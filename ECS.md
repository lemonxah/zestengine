# ZestEngine ECS Guide

How game logic is written in ZestEngine: **components** hold data, **systems** hold behaviour, and
**entities** are ready-made objects you can place in a scene or spawn from code. This guide explains
the model, then walks through the patterns that real games are made of: moving a player, taking
damage, spawning things, picking up coins, opening doors, shooting.

Every example in this guide is compiled and run by
[`crates/zestengine-core/tests/ecs_guide.rs`](https://github.com/lemonxah/zestengine/blob/main/crates/zestengine-core/tests/ecs_guide.rs), so the code
you copy works. Companion guides: [ENGINE_API.md](ENGINE_API.md) (what you can *do* from a system:
move, spawn, raycast, play sounds, …) and [WASM.md](WASM.md) (sandboxed scripts for user-generated
content). The generated API reference is at
[`api/zestengine_core`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/index.html).

## The big picture

ZestEngine's gameplay layer is an entity-component-system built on [hecs](https://crates.io/crates/hecs).
The vocabulary maps like this:

| Concept | In ZestEngine | Where |
|---|---|---|
| Entity | An **object** in the scene tree: a transform plus whatever components you attach. Saved with a stable `ObjectId` (UUID). | [`SceneWorld`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.SceneWorld.html) |
| Component | A plain Rust struct implementing `TypedComponent`: serializable data, shown in the inspector. | [`TypedComponent`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/trait.TypedComponent.html) |
| System | A function `fn(&SystemWorld, &mut SystemContext)` that queries components and changes them. | [`SystemFn`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/type.SystemFn.html) |
| Stage | The point in the frame a system runs: `Start`, `FixedUpdate`, `Update`, `Gameplay`, `LateUpdate`, `Events`. | [`Stage`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/enum.Stage.html) |
| Command | A request that changes the wider world (spawn, destroy, play a sound, load a scene). Queued by systems, applied at the end of the stage. | [`ComponentCommand`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/enum.ComponentCommand.html) |
| Event | A value any system can broadcast (`ctx.emit`) and any system can read next frame (`ctx.events`). | [`SystemContext`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.SystemContext.html) |
| Entity template | A named, ready-made object (mesh shape + starting components) registered by your project. | [`EntityTemplate`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.EntityTemplate.html) |

A word on naming. In ECS terms every scene object is an *entity*. In the **New Rust** menu, **Entity**
means the ready-made kind: a template such as "Enemy" or "Crate" that appears under *Create → Entity*
and can be spawned from code. This guide says **object** for the scene-tree thing and **entity** for the
template.

Three rules shape everything else:

1. **Data lives in components, behaviour lives in systems.** A component never has an `update`
   method that the engine calls. Systems find components by querying for them.
2. **Systems edit component data directly; they ask for everything else.** Changing a `Transform` or a
   `Health` is an ordinary write. Spawning, destroying, sounds, scene loads and physics pushes are
   queued as commands, so the world never changes underneath a running query.
3. **Systems run only while playing.** The editor's Play mode and a shipped game run the same loop. While
   you author a scene, nothing ticks.

## Objects

An object is a hecs entity carrying a fixed set of built-in components, plus the ones you attach:

| Built-in component | Meaning |
|---|---|
| `ObjectId` | Stable UUID. Saved in scenes, used by [`ObjectRef`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.ObjectRef.html) fields and the network. Survives reloads and reordering. |
| `Name` | The display name. Cosmetic: not unique. |
| `Transform` | **Local** translation / rotation / scale (relative to the parent). |
| `Parent` | The parent object, if any. |
| `Enabled` | The checkbox in the inspector. |
| `Active` | `Enabled` **and** every ancestor enabled. Computed; skip objects where it is false. |
| `GlobalTransform` | World matrix as of the last hierarchy propagation (a snapshot, refreshed at each stage boundary). |
| `Owner` | The object a component instance belongs to. Objects own themselves. |

Two details matter in practice.

**Local versus world.** `Transform` is local. When you need a true world position, ask the object context
(`obj.world_position()`, `obj.position_of(target)`) or `world.world_transform(entity)`, which folds in
every parent *including writes made earlier in the same stage*. `GlobalTransform` is only a snapshot.

**Duplicate components.** An object can carry two components of the same type. The first lives on the
object's own hecs entity; later ones live on small auxiliary entities that point back with `Owner`.
That is why systems query `(&Owner, &T)` rather than `&T`: `Owner` is how you get from a component
back to its object (for the transform, the `Active` flag, the `ObjectId`).

```rust
for (_, (owner, rotator)) in world.query::<(&Owner, &Rotator)>().iter() {
    let object: hecs::Entity = owner.0;            // the object this Rotator is on
    let id = world.object_id(object);              // its saved ObjectId
}
```

You almost never build objects by hand in game code. The editor builds them from scenes, and systems
spawn them with commands (see [Spawning and destroying](#spawning-and-destroying)). Tests use
`SceneWorld` directly (see [Testing systems](#testing-systems)).

## Components

A component is a struct that implements `TypedComponent`. It must be
`Serialize + Deserialize + Default + Send + Sync + 'static`.

```rust
use serde::{Deserialize, Serialize};
use zestengine_core::TypedComponent;

/// Spins the object around the Y axis.
#[derive(Serialize, Deserialize)]
pub struct Rotator {
    pub degrees_per_second: f32,
}

impl Default for Rotator {
    fn default() -> Self { Self { degrees_per_second: 45.0 } }
}

impl TypedComponent for Rotator {
    const NAME: &'static str = "Rotator";
}
```

Rules of thumb:

- **`NAME` is permanent.** It is the identity used by the Add Component menu, the `.scene` file and
  event bindings. Renaming it orphans saved data. If you must, keep the old one in `ALIASES`.
- **Saved versus runtime fields.** Everything that is not `#[serde(skip)]` is saved in the scene and
  shown in the inspector. Mark scratch state (timers, "have I started", cached positions)
  `#[serde(skip)]`; it keeps its value while playing and resets when a plugin reloads or a scene loads.
- **Vectors.** The workspace builds `glam` without its `serde` feature, so saved fields use arrays
  (`[f32; 3]`) and runtime-only fields use `Vec3`/`Quat` with `#[serde(skip)]`.
- **References to other objects** are [`ObjectRef`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.ObjectRef.html)
  fields (`ObjectRef::NONE` when unset). The inspector renders a drop target; at runtime you resolve it
  with `obj.get::<T>(self.target)`, `obj.position_of(self.target)`, and so on. They return `None` when
  the target is unset or was deleted, so always handle that case.
- **Add `#[serde(default)]`** to a component's struct when you expect to add fields later: old scenes
  keep loading.
- **Network role.** `const NET_ROLE` (default `Both`) says where a type exists: `Client` types are
  stripped from a dedicated server, `Server` types from players.
- **Disabled instances.** Switching a component off in the inspector moves its value into a
  `DisabledComponent<T>` column. Ordinary `T` queries (and `register_system_for::<T>`) see only enabled
  instances; the value, including runtime state, is kept for when it is switched back on.

Register the type so scenes can load it and systems can query it:

```rust
registry.register_ecs::<Rotator>();   // `register::<T>()` is the same call
```

### Actions and properties

Components can expose **actions** (functions) and **properties** (readable values) to the editor's
event bindings: "when the player enters this trigger, call *Door → Open*". Declare them as `static`
tables and implement three methods. This is also what makes a component callable from a
[WASM script](WASM.md).

```rust
use zestengine_core::{ActionCtx, ActionDesc, ActionType, ArgValue, ParamDesc, PropertyDesc};

static DOOR_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("Open", "Open", &[], true),
    ActionDesc::new("Close", "Close", &[], true),
    ActionDesc::new("SetSpeed", "Set Speed", &[ParamDesc::float("speed", 2.0)], true),
];
static DOOR_PROPERTIES: &[PropertyDesc] = &[PropertyDesc::new("open", ActionType::Bool)];

impl TypedComponent for Door {
    const NAME: &'static str = "Door";

    fn actions(&self) -> &'static [ActionDesc] { DOOR_ACTIONS }
    fn properties(&self) -> &'static [PropertyDesc] { DOOR_PROPERTIES }

    fn read_property(&self, property: &str) -> Option<ArgValue> {
        (property == "open").then_some(ArgValue::Bool(self.wants_open))
    }

    fn invoke_action(&mut self, _ctx: &mut ActionCtx<'_>, action: &str, args: &[ArgValue]) -> Result<(), String> {
        match action {
            "Open" => self.wants_open = true,
            "Close" => self.wants_open = false,
            "SetSpeed" => self.speed = args[0].as_f32().max(0.0),
            _ => return Err(zestengine_core::actions::unknown_action(Self::NAME, action)),
        }
        Ok(())
    }
}
```

- An action's **name is permanent** for the same reason `NAME` is; the label is what the editor shows.
- The last argument of `ActionDesc::new` is `ugc_safe`. Mark an action safe only if it cannot change
  scenes, hurt other players or leave the world. The editor's content-authoring dropdowns offer only
  safe actions, and sandboxed WASM modules in untrusted content can expose only safe ones; mods may
  additionally call only the actions the game's allow-list names. See
  [`ugc_safe` in the WASM Guide](WASM.md#ugc_safe) for exactly what is enforced where.
- Arguments arrive already coerced to the declared types and as long as the parameter list, so
  `args[0]` is safe to index.
- An action cannot read *itself* back through a property reader while it runs, but it can read other
  components, including other instances of its own type. `ctx.emit(event)` and `ctx.commands` work
  inside actions.

### Editor inspector and gizmo

A component's inspector UI and viewport gizmo live in the editor crate (`Inspect`, `Gizmo`) behind the
plugin's `editor` cargo feature, so a shipped game never links egui. The **New Rust → Component**
template generates both (see [Plugins in a project](#plugins-in-a-project)).

## Systems

A system is a plain function. It receives a read-only view of the world (components are borrowed
through it) and a per-frame context.

```rust
use glam::Quat;
use zestengine_core::Transform;
use zestengine_core::ecs::{Owner, Stage, SystemContext, SystemWorld};

fn rotate(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for (_, (owner, rotator)) in world.query::<(&Owner, &Rotator)>().iter() {
        if !ctx.is_active(world, owner.0) { continue; }
        let mut transform = world.get_mut::<Transform>(owner.0).expect("every object has a Transform");
        let step = Quat::from_rotation_y(rotator.degrees_per_second.to_radians() * ctx.dt);
        transform.rotation = (step * transform.rotation).normalize();
    }
}
```

Register it on the `ComponentRegistry`, together with its component:

```rust
pub fn register(registry: &mut ComponentRegistry) {
    registry.register_ecs::<Rotator>();
    registry.register_system_for::<Rotator>("game::rotate", Stage::Update, rotate);
}
```

| Registration | Runs the system |
|---|---|
| `register_system(name, stage, f)` | Every time the stage runs. |
| `register_system_for::<T>(name, stage, f)` | Only while at least one enabled `T` exists. Use this by default: unused features cost nothing. |
| `register_system_if(name, stage, f, condition)` | Only while `condition(world)` is true. Used for start-up work (see [Timers and state](#timers-and-state)). |

Names are unique. Registering an existing name replaces the code and stage in place (that is how a
hot-reloaded plugin supersedes its old build). The function type is a plain `fn` pointer, so a system
cannot capture variables: keep state in components, not in closures. (A non-capturing closure, like the
`|world, ctx| run_ordered(...)` forms below, coerces to a `fn`.)

### What a system can see

`SystemWorld` (the first argument):

| Method | Does |
|---|---|
| `query::<Q>()` | A hecs query, for example `(&Owner, &mut Health)`. |
| `get::<T>(entity)` / `get_mut::<T>(entity)` | One component of one entity, shared or exclusive. `None` if it has none. |
| `entity(id)` / `object_id(entity)` | Convert between a saved `ObjectId` and the runtime entity. |
| `world_transform(entity)` | The current world matrix, including parent writes made earlier in this stage. |
| `current_active(entity)` | Inherited activation right now. |

`SystemContext` (the second argument):

| Field / method | Meaning |
|---|---|
| `ctx.dt` | Seconds this frame, **scaled by the time scale** (0 while paused). In `FixedUpdate` it is the fixed step (1/60 s). |
| `ctx.unscaled_dt` | Ignores the time scale. Use it for pause menus and UI. |
| `ctx.time` | Scaled seconds since Play started. |
| `ctx.input` | The player's [actions](#input) this frame. |
| `ctx.commands` | The command queue (`push(ComponentCommand::…)`). |
| `ctx.emit(event)` / `ctx.events::<E>()` | Broadcast and read [events](#events). |
| `ctx.is_active(world, entity)` | `Active` and visible in the scene. Check it before touching an object. |
| `ctx.services` | Borrowed engine services: `physics`, `nav`, `net`, `screen`, `materials`, `save`, `strings`, `events`, `scene_events`, `components`, `order`. |

### The comfortable way: `run_ordered` and `ObjectContext`

Most gameplay code wants "for each active instance of my component, in scene order, with helpers for
this object". That is `run_ordered`, which gives your closure the component and an
[`ObjectContext`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.ObjectContext.html):
the per-object toolbox for moving, reading other objects, spawning, physics queries, sounds and more.
It also dereferences to the `SystemContext`, so `obj.dt`, `obj.input` and `obj.emit(..)` all work.

```rust
use glam::Vec3;
use zestengine_core::MotionContext;                       // translation(), set_translation(), world_position(), …
use zestengine_core::ecs::{ObjectContext, run_ordered};

#[derive(Serialize, Deserialize)]
pub struct PlayerMover {
    pub speed: f32,
}

impl PlayerMover {
    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        let direction = obj.input.axis2("Move"); // x = right, y = forward
        let step = Vec3::new(direction.x, 0.0, -direction.y) * self.speed * obj.dt;
        let position = obj.translation() + step;
        obj.set_translation(position);
    }
}

registry.register_system_for::<PlayerMover>("game::player_mover", Stage::Update, |world, ctx| {
    run_ordered(world, ctx, PlayerMover::update)
});
```

`run_ordered` does three things a hand-written loop does not:

- skips inactive objects (disabled, or under a disabled parent);
- iterates in **scene order**, so behaviour is deterministic. That matters for networking, prediction
  replay and anything where two objects interact within one frame. A bare `world.query` iterates in
  archetype order, which is an implementation detail;
- hands you an `ObjectContext` bound to the object.

Use a bare query when order is irrelevant and you have thousands of instances (it avoids collecting and
sorting); use `run_ordered` otherwise. The full method list of `ObjectContext` is in
[ENGINE_API.md](ENGINE_API.md).

### Borrowing rules

hecs checks borrows at run time, per component column, and the engine adds a lock for structural
changes. The rules that matter:

- **Two borrows of the same component type conflict if either is mutable.** `world.get_mut::<Transform>(e)`
  inside a query that already borrows `Transform` panics with `… already borrowed`. Nested queries are
  fine when they borrow different components (the *poison zone* example below).
- **No structural changes inside a system.** You cannot insert, remove or despawn components or
  objects through the world while a system runs. The world is locked, and the call panics with
  `structural ECS access while component views are borrowed`. Spawn and destroy with commands.
- **Use the `SystemWorld` you were given.** Editor-style component views
  (`SceneWorld::get`, `ComponentSet::get_typed`) panic inside a system with
  `component-view access during native ECS execution`.
- **A panic in a system is not swallowed.** Locks are released and component values stay intact, but the
  panic propagates, so treat `expect` like any other assertion.

When two of your own systems need the same data, run them in different stages or let one emit an event
for the other (see [Events](#events)).

## Stages

A frame runs the stages in this order. After each stage the engine applies the commands it queued.

```text
tick(dt)
 ├─ Start          repeats (up to 8×) while start-up work keeps spawning objects
 ├─ animation markers ──▶ Events stage (SceneEvent::Animation)
 ├─ FixedUpdate  ┐ 0 to 8 times per frame at a fixed 60 Hz, dt = 1/60
 │  physics step ┘
 ├─ contacts ───────────▶ Events stage (SceneEvent::Contact, 3D then 2D)
 ├─ Update
 ├─ Gameplay
 ├─ navigation agents move, animators evaluate
 └─ LateUpdate
```

| Stage | Use it for |
|---|---|
| `Start` | Initialising new instances: capture a rest position, cache a reference, spawn initial children. Not called "once" by the engine: guard with a flag (see [Timers and state](#timers-and-state)). |
| `FixedUpdate` | Continuous forces and anything that must match the physics rate. Runs before each physics step with `dt = 1/60`. |
| `Update` | Movement and general per-frame logic: controllers, AI steering, input handling. |
| `Gameplay` | Logic that must see movement already resolved: weapons, damage, pickups, health. Runs after `Update`'s movement commands (a character move, a teleport) have been applied. |
| `LateUpdate` | Things that read the final pose of the frame: cameras following a target, HUD bindings. Runs after animation and navigation. |
| `Events` | Reacting to contacts and animation markers through `ctx.services.scene_events`. Runs when such events exist. |

`ctx.dt` is `unscaled_dt × time_scale`, and the engine clamps a single frame to 0.25 s so a hitch
cannot teleport everything.

## Commands

Anything that reaches outside component data is a [`ComponentCommand`](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/enum.ComponentCommand.html)
pushed onto `ctx.commands`. The engine applies them, in order, right after the stage that queued them:

```rust
ctx.commands.push(ComponentCommand::Destroy(object_id));
ctx.commands.push(ComponentCommand::PlaySound { clip: "audio/pop.ogg".into(), position: None, volume: 1.0 });
```

`ObjectContext` wraps the common ones in readable methods (`obj.destroy(target)`,
`obj.play_sound_at(..)`, `obj.add_impulse(..)`), which is usually what you want. Two properties to
remember:

- **Queued means "after this stage".** `obj.destroy_self()` does not remove the object while your system
  is still iterating; it disappears when the stage ends. Systems later in the *same* stage still see it.
- **Bad targets are ignored.** A command that names an object that no longer exists does nothing.

Direct writes to component data (`obj.edit::<T>(target, |t| ..)`, `obj.set_translation(..)`) are
immediate and visible to systems that run later in the same stage. The full list of commands and which
`ObjectContext` methods queue them is in [ENGINE_API.md](ENGINE_API.md).

## Events

There are three ways for systems to react to something that happened.

### Broadcast events

Any `'static` type is an event. A system emits one; **every** system can read it **next frame**. The
sender and the receiver never touch each other's data, which is the whole point.

```rust
/// Event: something hit `target` for `amount`.
pub struct Hit { pub target: ObjectId, pub amount: f32 }
/// Event: the object ran out of hitpoints.
pub struct Died { pub object: ObjectId }

#[derive(Serialize, Deserialize)]
pub struct Hitpoints { pub current: f32, pub max: f32 }   // Default: 100 / 100

fn apply_hits(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    // Events emitted during the previous frame are readable now.
    let hits: Vec<&Hit> = ctx.events::<Hit>().collect();
    if hits.is_empty() { return; }
    for (_, (owner, hp)) in world.query::<(&Owner, &mut Hitpoints)>().iter() {
        let Some(me) = world.object_id(owner.0) else { continue };
        let before = hp.current;
        for hit in hits.iter().filter(|hit| hit.target == me) {
            hp.current = (hp.current - hit.amount).max(0.0);
        }
        if before > 0.0 && hp.current == 0.0 {
            ctx.emit(Died { object: me }); // readable next frame
        }
    }
}

fn remove_the_dead(_: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for died in ctx.events::<Died>() {
        ctx.commands.push(ComponentCommand::Destroy(died.object));
    }
}

registry.register_system_for::<Hitpoints>("game::apply_hits", Stage::Gameplay, apply_hits);
registry.register_system("game::remove_the_dead", Stage::Gameplay, remove_the_dead);
```

An event emitted in frame *N* is readable throughout frame *N + 1*; an event you emit is **not** visible
to you in the same frame. The engine's own systems use the same bus: weapons emit `DamageEvent`, health
emits `DeathEvent`, target dummies emit `ScoreEvent`, animators emit `AnimatorStateChanged` and
`AnimationEvent`, and the HUD reads them. Subscribing to those from your own systems is just
`ctx.events::<DamageEvent>()`.

### Scene events: contacts and animation markers

Collisions, trigger overlaps and animation markers arrive **within the frame** in the `Events` stage,
as `ctx.services.scene_events`:

```rust
use zestengine_core::ecs::{ContactPhase, SceneEvent};

#[derive(Default, Serialize, Deserialize)]
pub struct Collector {}                       // marker: can pick coins up

#[derive(Serialize, Deserialize)]
pub struct Coin { pub value: u32 }            // Default: value 1

/// Event: `by` picked up something worth `value`.
pub struct Collected { pub by: ObjectId, pub value: u32 }

registry.register_system_for::<Coin>("game::coin_pickup", Stage::Events, |world, ctx| {
    let contacts = ctx.services.scene_events;
    run_ordered(world, ctx, |coin: &mut Coin, obj| {
        let me = obj.self_id();
        for contact in contacts {
            if let SceneEvent::Contact { object, other, trigger: true, phase: ContactPhase::Enter, .. } = contact
                && *object == me
                && obj.get::<Collector>(*other).is_some()
            {
                obj.emit(Collected { by: *other, value: coin.value });
                obj.destroy_self();
            }
        }
    })
});
```

Both objects of a touching pair receive a `Contact` event (`object` is the receiver, `other` the one it
touched), with `phase` `Enter`, `Stay` (2D) or `Exit`, `trigger` telling a trigger overlap from a
solid collision, and `two_dimensional` for 2D physics. The coin needs a collider with *Is Trigger*
on; the player needs a collider and a rigid body or character.

### Action bindings

The third way is data-driven: the inspector's *Trigger Events*, *Interactable*, *UI Button* and
*Animation Event Bindings* components store lists of "call this action on that component" and the
engine resolves and runs them. You get that for free for any component that declares
[actions](#actions-and-properties). Gameplay that fires a binding from code uses
`queue_actions(&bindings, &FireSource { … }, ctx.commands)`.

## Spawning and destroying

Spawning is a command; the call returns the new `ObjectId` immediately so you can wire it up, but the
object exists from the next stage on.

```rust
let id = obj.spawn_prefab("prefabs/enemy.prefab", position, rotation); // a .prefab asset
let id = obj.instantiate(self.bullet_template, position, rotation);    // clone a scene object
let id = obj.spawn_entity("Crate", position, rotation);                // a registered entity
obj.net_spawn("prefabs/bullet.prefab", position, rotation);            // on every peer
obj.destroy(target);                                                    // object and children
obj.destroy_self();
obj.set_active(door, false);                                            // disable without deleting
```

A timed self-destruct is the smallest useful spawn companion:

```rust
#[derive(Serialize, Deserialize)]
pub struct Lifetime { pub seconds_left: f32 }   // Default: 3.0

impl Lifetime {
    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        self.seconds_left -= obj.dt;
        if self.seconds_left <= 0.0 {
            obj.destroy_self();
        }
    }
}
```

Objects spawned during a frame run their `Start` systems before the next `Update`, so a freshly
spawned object is initialised before it acts. See [Entities](#entities) for a spawner that drops
crates.

## Entities

An **entity** is a named, ready-made object: a mesh shape plus the components it starts with. Register
it once; it then appears under **Create → Entity** in the editor and any system can spawn it by name.

```rust
use zestengine_core::ecs::{EntityShape, EntityTemplate};

/// The entity: a cube that deletes itself after ten seconds.
fn crate_entity() -> EntityTemplate {
    EntityTemplate::new(EntityShape::Cube).with(Lifetime { seconds_left: 10.0 })
}

registry.register_entity("Crate", crate_entity);
```

- `EntityShape` is `Empty`, `Cube`, `Sphere`, `Capsule` or `Plane`. `Empty` makes a transform that only
  carries components (a spawner, a manager); the others also get a `MeshRenderer` with the default
  material.
- `.with(component)` adds a component. Add as many as you like, in inspector order. The builder function
  runs for **every** new instance, so each object gets its own component values.
- The name is the identity: `spawn_entity("Crate", ..)` and the menu both use it. Registering an
  existing name replaces the builder, and a plugin reload drops entities the plugin no longer exports.
- An entity made in the editor is an ordinary scene object once placed: it saves as its components
  and mesh, and you can edit it freely.
- On a dedicated server (no renderer) an entity still spawns, with its components and without a GPU
  mesh.

A spawner is just a component and a system:

```rust
#[derive(Serialize, Deserialize)]
pub struct CrateSpawner {
    pub every: f32,
    #[serde(skip)]
    pub timer: f32,
}

impl CrateSpawner {
    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        self.timer -= obj.dt;
        if self.timer <= 0.0 {
            self.timer = self.every;
            let above = obj.world_position() + Vec3::Y;
            obj.spawn_entity("Crate", above, Quat::IDENTITY);
        }
    }
}
```

## Talking to other objects

Systems often need another object's data: a follower needs its target's position, a turret needs its
target's health. Reference the other object with an `ObjectRef` field and ask the `ObjectContext`.

```rust
#[derive(Serialize, Deserialize)]
pub struct Follower {
    pub target: ObjectRef,
    pub speed: f32,
}

impl Follower {
    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        let Some(goal) = obj.position_of(self.target) else { return }; // world position
        let here = obj.world_position();
        let to_goal = goal - here;
        let step = self.speed * obj.dt;
        if to_goal.length() <= step {
            obj.set_world_position(goal);
        } else {
            obj.set_world_position(here + to_goal.normalize() * step);
        }
    }
}
```

| Want to … | Call | Timing |
|---|---|---|
| Read another object's component | `obj.get::<T>(target)` | immediate |
| Change it in place | `obj.edit::<T>(target, \|t\| …)` | immediate (visible to later systems this stage) |
| Read its world position | `obj.position_of(target)` | immediate |
| Find an object by name | `obj.find_object("Player")` | immediate (names are cosmetic: prefer `ObjectRef`) |
| List objects carrying a component | `obj.objects_with::<T>()` | immediate, in scene order |
| Ask the world to do something to it | `obj.destroy(target)`, `obj.set_active(..)`, `obj.add_impulse(..)` | after the stage |

`get` and `edit` return `None` / `false` for an unset, deleted or hidden target; handle it.

When one component should act on **every** object of another kind, query both. This swamp hurts
anything with `Hitpoints` inside its radius; the nested query is legal because the two loops borrow
different components:

```rust
fn poison(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for (_, (owner, zone)) in world.query::<(&Owner, &PoisonZone)>().iter() {
        if !ctx.is_active(world, owner.0) { continue; }
        let Some(center) = world.world_transform(owner.0).map(|m| m.w_axis.truncate()) else { continue };
        for (_, (victim, hp)) in world.query::<(&Owner, &mut Hitpoints)>().iter() {
            let Some(at) = world.world_transform(victim.0).map(|m| m.w_axis.truncate()) else { continue };
            if at.distance(center) <= zone.radius {
                hp.current = (hp.current - zone.damage_per_second * ctx.dt).max(0.0);
            }
        }
    }
}
```

## Input

Input is a snapshot of **actions** (named, rebindable), not raw keys: `ctx.input` is an `InputState`.

| Call | Meaning |
|---|---|
| `input.down("Fire")` | Held this frame. |
| `input.pressed("Fire")` | Went down this frame (edge). |
| `input.released("Fire")` | Went up this frame (edge). |
| `input.axis("Throttle")` | 1D value, −1 to 1 (0/1 for a button). |
| `input.axis2("Move")` | 2D value. `x` is right, `y` is forward. |

The default keyboard-and-mouse scheme defines `Move` (WASD), `Look` (mouse), `Jump`, `Sprint`, `Crouch`,
`Fire`, `Interact`, `Grab` and `Voice`; the gamepad scheme uses the same names on its sticks and
buttons, so code written against action names works on every device. Players can rebind them, and a
game can define its own actions in its control scheme. An action that does not exist reads as "not
pressed". When the player is typing into a UI text field, `obj.ui_text_editing()` is true: skip
gameplay input then.

For networked objects another peer may be driving, check `obj.driven_remotely()` before applying
local input, and `obj.has_authority(id)` before changing shared state.

## Physics and contacts

Physics queries are immediate; pushes are commands. A ray gun:

```rust
#[derive(Serialize, Deserialize)]
pub struct Laser { pub damage: f32, pub range: f32 }   // Default: 25 / 100

impl Laser {
    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        if !obj.input.pressed("Fire") { return; }
        let aim = obj.self_transform(); // world position and orientation
        let Some(hit) = obj.raycast(aim.translation, aim.forward(), self.range) else { return };
        let damage = self.damage;
        obj.edit::<Hitpoints>(hit.object, move |hp| hp.current = (hp.current - damage).max(0.0));
        obj.add_impulse(hit.object, aim.forward() * 5.0);
        obj.play_sound_at("audio/zap.ogg", hit.point, 1.0);
    }
}
```

`obj.raycast` ignores triggers and the caller's own colliders; use `raycast_filtered` with a
`RayFilter` for layer masks. Rigid bodies move through commands (`add_force`, `add_impulse`,
`set_velocity`); characters through `move_character`. Contacts arrive as [scene events](#scene-events-contacts-and-animation-markers).
The ENGINE_API guide covers [physics](ENGINE_API.md#physics) in full.

## Timers and state

Three patterns cover most state.

**A cooldown** is a `#[serde(skip)]` float you count down, as in `CrateSpawner::timer` above.

**A state machine** is an enum field. Derive `Serialize, Deserialize` on the enum and give the
component a `Default`; match on it in the system and change it as conditions are met.

**Start-up work** uses the `Start` stage. The engine does not call "start once per instance"; the
stage simply runs before `Update` each frame, and your component remembers whether it has initialised.
Register the system with `register_system_if` so it costs nothing once every instance has started:

```rust
#[derive(Serialize, Deserialize)]
pub struct Door {
    pub lift: f32,
    pub speed: f32,
    #[serde(skip)]
    pub wants_open: bool,
    #[serde(skip)]
    pub closed_y: Option<f32>,        // None until started
}

impl Door {
    /// Runs once for each new instance (Stage::Start): remember where "closed" is.
    fn start(&mut self, obj: &mut ObjectContext<'_, '_>) {
        if self.closed_y.is_none() {
            self.closed_y = Some(obj.translation().y);
        }
    }

    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        let Some(closed) = self.closed_y else { return };
        let goal = if self.wants_open { closed + self.lift } else { closed };
        let mut position = obj.translation();
        let step = self.speed * obj.dt;
        position.y += (goal - position.y).clamp(-step, step);
        obj.set_translation(position);
    }
}

registry.register_system_if(
    "game::door_start",
    Stage::Start,
    |world, ctx| run_ordered(world, ctx, Door::start),
    |world| world.query::<&Door>().iter().any(|(_, door)| door.closed_y.is_none()),
);
registry.register_system_for::<Door>("game::door", Stage::Update, |world, ctx| {
    run_ordered(world, ctx, Door::update)
});
```

The `Door` of the [actions](#actions-and-properties) example is this same component: *Open* and
*Close* set `wants_open`, and any trigger, button or WASM script can bind to them.

### Showing state in the UI

UI elements are components too, so a system updates them like any other data:

```rust
/// Shows the score in this object's `UiText`.
#[derive(Default, Serialize, Deserialize)]
pub struct ScoreBoard {
    #[serde(skip)]
    pub score: u32,
}

impl ScoreBoard {
    fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
        for collected in obj.events::<Collected>() {
            self.score += collected.value;
        }
        let id = obj.self_id();
        let text = format!("Score {}", self.score);
        obj.edit::<UiText>(id, move |ui| ui.text = text);
    }
}

registry.register_system_for::<ScoreBoard>("game::score_board", Stage::LateUpdate, |world, ctx| {
    run_ordered(world, ctx, ScoreBoard::update)
});
```

`LateUpdate` is the right stage for anything that mirrors the frame's final state onto the screen.

## Plugins in a project

Your game's components, systems and entities live in a Rust crate, `plugins/components`, that the
editor builds and hot-reloads and a shipped game links. **Files panel → Create → New Rust** adds to it:

| Menu item | Writes | Registers with |
|---|---|---|
| **New Rust → Component** | `src/components/new_component.rs`: data struct, `TypedComponent` impl, inspector and gizmo stubs | `registry.register_ecs::<T>()` and the editor traits |
| **New Rust → System** | `src/systems/new_system.rs`: a system function with a query skeleton | `registry.register_system(..)` |
| **New Rust → Entity** | `src/entities/new_entity.rs`: an `EntityTemplate` builder | `registry.register_entity(..)` |

Each press creates the next free name (`new_component`, `new_component_1`, …), opens the file, and wires
it into `lib.rs`. A fresh plugin crate looks like this (the `orbit` and `orbiter` files are a working
starter that shows the three kinds side by side):

```text
plugins/components/
├── Cargo.toml
└── src/
    ├── lib.rs                 ← consolidates everything into the game
    ├── components/orbit.rs    ← data: the Orbit component (+ inspector)
    ├── systems/orbit.rs       ← behaviour: moves every Orbit
    └── entities/orbiter.rs    ← a ready-made sphere that carries an Orbit
```

`lib.rs` is the one place that hands every component, system and entity to the engine:

```rust
mod components {
    pub mod orbit;
    // zestengine: new component modules here
}
mod systems {
    pub mod orbit;
    // zestengine: new system modules here
}
mod entities {
    pub mod orbiter;
    // zestengine: new entity modules here
}

zestengine_core::export_plugin_abi!();

#[unsafe(no_mangle)]
pub fn zestengine_register(registry: &mut ComponentRegistry) {
    // Components: data attached to objects.
    components::orbit::register(registry);
    // zestengine: new components register here

    // Systems: behaviour that runs every frame over component data.
    systems::orbit::register(registry);
    // zestengine: new systems register here

    // Entities: ready-made objects (Create > Entity, `spawn_entity`).
    entities::orbiter::register(registry);
    // zestengine: new entities register here
}
```

The `// zestengine:` comments are anchors the menu inserts above; leave them in place. Each module
exports its own `register` function, so `lib.rs` never names the types inside: rename a struct freely,
and when you rename or move a *file* just update its two lines in `lib.rs`. A plugin crate made before
this layout keeps working: the menu falls back to flat modules next to `lib.rs` and registers them at
the top of `zestengine_register`.

How a plugin gets into the game:

- **In the editor**, **Tools → Build & Reload Components** compiles the crate with its `editor` feature
  and registers the new build over the old one. Component values are re-created from their serialized
  state, so `#[serde(skip)]` runtime fields reset on reload.
- **In a shipped game**, `src/main.rs` calls `zestengine_project_components::zestengine_register(registry)`
  from `run_game`, linking the plugin statically (the editor feature, `egui` and the editor crate are
  not linked).
- **On a dedicated server**, the built `cdylib` is listed in `server.ron` and loaded at start-up.
- **ABI stamp.** `export_plugin_abi!()` lets a host refuse a plugin built with a different rustc, engine
  version or dependency features instead of crashing. Rebuild plugins and the engine together.

## Testing systems

Systems are plain functions over a world, so they test without a window, a GPU or the engine loop. Build
a `SceneWorld`, register your component and systems on a `ComponentRegistry`, run a stage, and look at
the data and the queued commands. This is the harness the examples in this guide use:

```rust
use zestengine_core::ecs::{SceneEvent, SceneWorld, Stage, SystemContext};
use zestengine_core::{AnyBox, ComponentCommand, ComponentRegistry, InputState};

#[derive(Default)]
struct Frame {
    dt: f32,
    input: InputState,
    events: Vec<AnyBox>,
    scene_events: Vec<SceneEvent>,
}

/// Run every system registered for `stage` once and return the commands they queued.
fn run(registry: &ComponentRegistry, scene: &SceneWorld, stage: Stage, frame: &Frame) -> Vec<ComponentCommand> {
    let mut commands = Vec::new();
    let mut ctx = SystemContext::new(&frame.input, &mut commands);
    ctx.dt = frame.dt;
    ctx.services.events = &frame.events;
    ctx.services.scene_events = &frame.scene_events;
    registry.run_systems(stage, scene, &mut ctx);
    commands
}

#[test]
fn spins_active_objects_only() {
    let mut registry = ComponentRegistry::default();
    register(&mut registry);
    let mut scene = SceneWorld::new();
    let spinning = scene.spawn("Spinning");
    scene.insert(spinning, Rotator { degrees_per_second: 90.0 }).unwrap();
    run(&registry, &scene, Stage::Update, &Frame { dt: 1.0, ..Default::default() });
    let expected = Quat::from_rotation_y(90f32.to_radians());
    assert!(scene.get::<Transform>(spinning).unwrap().rotation.abs_diff_eq(expected, 1e-5));
}
```

Useful pieces:

- `SceneWorld::spawn`, `insert`, `get`, `get_mut`, `remove`, `set_parent`, `propagate` build and inspect
  scenes. After changing a parent or `Enabled` flag, call `propagate()` so `Active` and
  `GlobalTransform` update, as the engine does at stage boundaries.
- **Events**: set `ctx.services.events = &[AnyBox(Box::new(Hit { … }))]` to simulate last frame's
  broadcasts, and `ctx.services.scene_events` for contacts. Read what your system emitted from
  `commands` (`ComponentCommand::Emit`).
- **Services**: set `ctx.services.physics = Some(&mock)` with your own `PhysicsQuery` to fake raycasts
  (the `Laser` test does this), `nav`, `screen` and `net` likewise.
- **Actions**: call `component.invoke_action(&mut ActionCtx { … }, "Open", &[])` directly.
- Multi-frame behaviour: call `run` in a loop and assert after each step; commands queued by one frame
  are not applied by the harness, so assert on them instead.

## Common mistakes

| Symptom | Cause and fix |
|---|---|
| Panic: `… already borrowed` | Two borrows of one component type, one mutable (often `get_mut::<Transform>` inside a query that has `&mut Transform`). Query different components, or collect first. |
| Panic: `structural ECS access while component views are borrowed` | Inserting, removing or despawning inside a system. Queue a command. |
| System never runs | Not registered (add it to `zestengine_register`), registered with `register_system_for::<T>` while no enabled `T` exists, or the object is inactive (disabled or under a disabled parent). |
| Component missing from the Add menu or lost from scenes | `register_ecs::<T>()` not called, or `NAME` changed. |
| Object is in the wrong place under a parent | A world position written to the local `Transform`. Use `set_world_position` / `set_world_position_of`. |
| Spawned object not found next line | Spawns are commands; it exists from the next stage. Keep the returned `ObjectId`. |
| Event never arrives | Events emitted in frame *N* are readable in *N + 1*, never the same frame. Check the type is exactly the one emitted. |
| Behaviour differs per frame rate | Multiplying by a constant instead of `ctx.dt`. |
| Objects react in a different order each run | Iterating with a bare `query` where order matters. Use `run_ordered`. |
| Runtime state lost after a reload | It is `#[serde(skip)]` and plugins re-create components on reload. Expected. |

## Reference

| Item | API page |
|---|---|
| `TypedComponent` | [trait.TypedComponent](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/trait.TypedComponent.html) |
| `ComponentRegistry` (`register_ecs`, `register_system*`, `register_entity`) | [struct.ComponentRegistry](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.ComponentRegistry.html) |
| `SystemWorld`, `SystemContext`, `SystemServices` | [ecs/struct.SystemWorld](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.SystemWorld.html), [ecs/struct.SystemContext](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.SystemContext.html), [ecs/struct.SystemServices](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.SystemServices.html) |
| `Stage`, `SceneEvent`, `ContactPhase` | [ecs/enum.Stage](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/enum.Stage.html), [ecs/enum.SceneEvent](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/enum.SceneEvent.html) |
| `run_ordered`, `ObjectContext` | [ecs/fn.run_ordered](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/fn.run_ordered.html), [ecs/struct.ObjectContext](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.ObjectContext.html) |
| `SceneWorld` | [ecs/struct.SceneWorld](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.SceneWorld.html) |
| `EntityTemplate`, `EntityShape` | [ecs/struct.EntityTemplate](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/struct.EntityTemplate.html), [ecs/enum.EntityShape](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/ecs/enum.EntityShape.html) |
| `ComponentCommand` | [enum.ComponentCommand](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/enum.ComponentCommand.html) |
| `ObjectId`, `ObjectRef`, `Transform` | [struct.ObjectId](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.ObjectId.html), [struct.ObjectRef](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.ObjectRef.html), [struct.Transform](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.Transform.html) |
| `InputState` | [struct.InputState](https://lemonxah.github.io/zestengine/docs/api/zestengine_core/struct.InputState.html) |

Where to go next: [ENGINE_API.md](ENGINE_API.md) for everything a system can do to the world,
[WASM.md](WASM.md) for scripting that runs in a sandbox, and the built-in gameplay kit
(`crates/zestengine-core/src/kit/`: health, hitscan weapons, character controllers, RTS selection, VR
rigs) for larger, production examples written with exactly the patterns above.
