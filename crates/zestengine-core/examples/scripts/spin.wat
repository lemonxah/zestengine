;; Tiny Script component example: slide the owning object along X each frame
;; while counting elapsed time in property slot 0.
;;
;; The engine hosts this module with the `env` import namespace below. No
;; WASI imports are linked, so the module has no file-system or network access.
;;
;; Exports:
;;   start  - run once when the object becomes active
;;   update - run once per frame
;;
;; Host API (all floats are little-endian IEEE 754):
;;   env.get_dt() -> f32
;;   env.get_time() -> f32
;;   env.set_position(x, y, z)
;;   env.get_position() -> (x, y, z)
;;   env.set_rotation(x, y, z, w)
;;   env.get_rotation() -> (x, y, z, w)
;;   env.set_scale(x, y, z)
;;   env.get_scale() -> (x, y, z)
;;   env.get_prop(index: i32) -> f32
;;   env.set_prop(index: i32, value: f32)

(module
  (import "env" "get_dt" (func $get_dt (result f32)))
  (import "env" "set_position" (func $set_position (param f32 f32 f32)))
  (import "env" "get_position" (func $get_position (result f32 f32 f32)))
  (import "env" "set_rotation" (func $set_rotation (param f32 f32 f32 f32)))
  (import "env" "set_scale" (func $set_scale (param f32 f32 f32)))
  (import "env" "get_prop" (func $get_prop (param i32) (result f32)))
  (import "env" "set_prop" (func $set_prop (param i32 f32)))

  ;; One-time setup: lift the object up and give it a distinctive scale.
  (func (export "start")
    f32.const 0.0
    f32.const 1.0
    f32.const 0.0
    call $set_position

    f32.const 1.0
    f32.const 2.0
    f32.const 3.0
    call $set_scale

    i32.const 0
    f32.const 0.0
    call $set_prop)

  ;; Per-frame: read the current position, advance X by 2 * dt, and write the
  ;; transform back; props[0] accumulates dt.
  (func (export "update")
    (local $x f32)
    (local $y f32)
    (local $z f32)
    (local $v f32)

    ;; get_position returns (x, y, z) with x pushed first, so z is on top.
    call $get_position
    local.set $z
    local.set $y
    local.set $x

    ;; x = x + 2 * dt
    local.get $x
    call $get_dt
    f32.const 2.0
    f32.mul
    f32.add
    local.set $x

    local.get $x
    local.get $y
    local.get $z
    call $set_position

    ;; A half-turn around Y (still a unit quaternion).
    f32.const 0.0
    f32.const 1.0
    f32.const 0.0
    f32.const 0.0
    call $set_rotation

    ;; props[0] += dt
    i32.const 0
    call $get_prop
    local.set $v

    i32.const 0
    local.get $v
    call $get_dt
    f32.add
    call $set_prop))
