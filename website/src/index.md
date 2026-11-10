# ZestEngine Documentation

ZestEngine is a Vulkan game engine written in Rust, with a built-in editor, VR support, realtime
global illumination, physics, networking and a sandboxed WebAssembly scripting layer. Gameplay code is
plain Rust: **components** hold data, **systems** hold behaviour, and **entities** are ready-made objects
you place in a scene or spawn from code.

## Start here

| If you want to … | Read |
|---|---|
| Understand components, systems, stages, events and entities, with working examples | [ECS Guide](ECS.md) |
| Know how to move things, spawn things, raycast, play sounds, read input, save and network | [Engine API Guide](ENGINE_API.md) |
| Run sandboxed scripts for user-generated content and mods | [WASM Guide](WASM.md) |
| Look up an exact type or function | [Rust API Reference](api.md) |

## How the guides fit together

1. The **[ECS Guide](ECS.md)** teaches the model. Every example in it is compiled and run by the
   engine's test suite, so the code you copy works.
2. The **[Engine API Guide](ENGINE_API.md)** is organised by task ("moving things", "spawning and
   destroying", "physics") and says for each operation whether it takes effect immediately or when the
   stage ends.
3. The **[WASM Guide](WASM.md)** covers the second way to write gameplay: modules compiled to
   WebAssembly that run in a sandbox, for content you do not fully trust.
4. The **[Rust API Reference](api.md)** is generated from the doc comments in the source. Every item
   links back to the matching guide section.

## Your first system in 30 seconds

In the editor, open the Files panel, right-click, and choose **Create → New Rust → Component**, then
**New Rust → System**. The engine writes starter files into your project's `plugins/components` crate and
registers them in its `lib.rs`. Put data in the component, behaviour in the system:

```rust
fn rotate(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for (_, (owner, rotator)) in world.query::<(&Owner, &Rotator)>().iter() {
        if !ctx.is_active(world, owner.0) { continue; }
        let mut transform = world.get_mut::<Transform>(owner.0).expect("every object has a Transform");
        let step = Quat::from_rotation_y(rotator.degrees_per_second.to_radians() * ctx.dt);
        transform.rotation = (step * transform.rotation).normalize();
    }
}
```

Then **Tools → Build & Reload Components**, add the component to an object and press Play. The
[ECS Guide](ECS.md) explains every line.

## Source

ZestEngine's source lives at [github.com/lemonxah/zestengine](https://github.com/lemonxah/zestengine).
