// One-call ASTC HDR encode over the vendored ARM astcenc library, so the Rust
// side needs no bindings to astcenc's config structs.
#include <cstddef>
#include <cstdint>

#include "astcenc/astcenc.h"

// Encodes a `w` x `h` RGBA half-float image into ASTC HDR blocks of
// `bx` x `by`. `out` must hold ceil(w/bx) * ceil(h/by) * 16 bytes.
// Returns 0 on success.
extern "C" int zestengine_astc_hdr_encode(const uint16_t* rgba_f16, uint32_t w, uint32_t h, uint32_t bx, uint32_t by,
                                      uint8_t* out, size_t out_len) {
    astcenc_config config;
    if (astcenc_config_init(ASTCENC_PRF_HDR, bx, by, 1, ASTCENC_PRE_MEDIUM, 0, &config) != ASTCENC_SUCCESS) {
        return 1;
    }
    astcenc_context* context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
        return 2;
    }
    void* slices[1] = {const_cast<uint16_t*>(rgba_f16)};
    astcenc_image image{w, h, 1, ASTCENC_TYPE_F16, slices};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const astcenc_error result = astcenc_compress_image(context, &image, &swizzle, out, out_len, 0);
    astcenc_context_free(context);
    return result == ASTCENC_SUCCESS ? 0 : 3;
}

// Decodes ASTC HDR blocks back to RGBA half floats (tests check the encoder
// with it). `rgba_f16` must hold w * h * 4 values. Returns 0 on success.
extern "C" int zestengine_astc_hdr_decode(const uint8_t* blocks, size_t blocks_len, uint32_t w, uint32_t h, uint32_t bx,
                                      uint32_t by, uint16_t* rgba_f16) {
    astcenc_config config;
    if (astcenc_config_init(ASTCENC_PRF_HDR, bx, by, 1, ASTCENC_PRE_MEDIUM, ASTCENC_FLG_DECOMPRESS_ONLY, &config) !=
        ASTCENC_SUCCESS) {
        return 1;
    }
    astcenc_context* context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
        return 2;
    }
    void* slices[1] = {rgba_f16};
    astcenc_image image{w, h, 1, ASTCENC_TYPE_F16, slices};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    const astcenc_error result = astcenc_decompress_image(context, blocks, blocks_len, &image, &swizzle, 0);
    astcenc_context_free(context);
    return result == ASTCENC_SUCCESS ? 0 : 3;
}

// Encodes a `w` x `h` RGBA8 image into ASTC LDR blocks of `bx` x `by`
// (partial edge blocks included). Without `alpha` the alpha channel is
// encoded as opaque. `out` must hold ceil(w/bx) * ceil(h/by) * 16 bytes.
// Returns 0 on success.
extern "C" int zestengine_astc_ldr_encode(const uint8_t* rgba, uint32_t w, uint32_t h, uint32_t bx, uint32_t by, int alpha,
                                      uint8_t* out, size_t out_len) {
    astcenc_config config;
    if (astcenc_config_init(ASTCENC_PRF_LDR, bx, by, 1, ASTCENC_PRE_MEDIUM, 0, &config) != ASTCENC_SUCCESS) {
        return 1;
    }
    astcenc_context* context = nullptr;
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
        return 2;
    }
    void* slices[1] = {const_cast<uint8_t*>(rgba)};
    astcenc_image image{w, h, 1, ASTCENC_TYPE_U8, slices};
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, alpha ? ASTCENC_SWZ_A : ASTCENC_SWZ_1};
    const astcenc_error result = astcenc_compress_image(context, &image, &swizzle, out, out_len, 0);
    astcenc_context_free(context);
    return result == ASTCENC_SUCCESS ? 0 : 3;
}
