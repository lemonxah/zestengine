use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ash::vk;

static REAL_PROC_ADDR: AtomicUsize = AtomicUsize::new(0);
static FAILURES: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn hold(text: String) {
    let mut held = FAILURES.lock().unwrap_or_else(|e| e.into_inner());
    if held.len() < 12 {
        held.push(text);
    }
}

fn note(text: String) {
    tracing::error!("FFX backend: {text}");
    hold(text);
}

fn absent(text: String) {
    tracing::debug!("FFX backend: {text}");
    hold(text);
}

pub fn take() -> String {
    let mut held = FAILURES.lock().unwrap_or_else(|e| e.into_inner());
    let text = held.join("; ");
    held.clear();
    text
}

pub fn install(real: vk::PFN_vkGetDeviceProcAddr) -> *mut c_void {
    REAL_PROC_ADDR.store(real as *const () as usize, Ordering::Relaxed);
    proc_addr as *const () as *mut c_void
}

const DEBUG_UTILS: [&[u8]; 3] = [b"vkCmdBeginDebugUtilsLabelEXT", b"vkCmdEndDebugUtilsLabelEXT", b"vkSetDebugUtilsObjectNameEXT"];

type Create = unsafe extern "system" fn(vk::Device, *const c_void, *const c_void, *mut c_void) -> vk::Result;

macro_rules! traced_create {
    ($($slot:ident, $tramp:ident, $name:literal;)*) => {
        $(
            static $slot: AtomicUsize = AtomicUsize::new(0);
            unsafe extern "system" fn $tramp(device: vk::Device, info: *const c_void, allocator: *const c_void, out: *mut c_void) -> vk::Result {
                let real = unsafe { std::mem::transmute::<usize, Create>($slot.load(Ordering::Relaxed)) };
                let result = unsafe { real(device, info, allocator, out) };
                if result != vk::Result::SUCCESS {
                    note(format!("{} returned {result:?}{}", $name, unsafe { detail($name, info) }));
                }
                result
            }
        )*
        fn create_trampoline(name: &[u8], real: usize) -> Option<usize> {
            $(
                if name == $name.as_bytes() {
                    $slot.store(real, Ordering::Relaxed);
                    return Some($tramp as *const () as usize);
                }
            )*
            None
        }
    };
}

traced_create! {
    CREATE_IMAGE, trace_create_image, "vkCreateImage";
    CREATE_BUFFER, trace_create_buffer, "vkCreateBuffer";
    ALLOCATE_MEMORY, trace_allocate_memory, "vkAllocateMemory";
    CREATE_IMAGE_VIEW, trace_create_image_view, "vkCreateImageView";
    CREATE_SAMPLER, trace_create_sampler, "vkCreateSampler";
    CREATE_SHADER_MODULE, trace_create_shader_module, "vkCreateShaderModule";
    CREATE_SET_LAYOUT, trace_create_set_layout, "vkCreateDescriptorSetLayout";
    CREATE_PIPELINE_LAYOUT, trace_create_pipeline_layout, "vkCreatePipelineLayout";
    CREATE_DESCRIPTOR_POOL, trace_create_descriptor_pool, "vkCreateDescriptorPool";
    CREATE_RENDER_PASS, trace_create_render_pass, "vkCreateRenderPass";
    CREATE_COMMAND_POOL, trace_create_command_pool, "vkCreateCommandPool";
}

unsafe fn detail(name: &str, info: *const c_void) -> String {
    if info.is_null() {
        return String::new();
    }
    match name {
        "vkCreateImage" => {
            let info = unsafe { &*(info as *const vk::ImageCreateInfo<'_>) };
            format!(" (format {:?}, usage {:?}, extent {}x{}x{}, mips {}, flags {:?})", info.format, info.usage, info.extent.width, info.extent.height, info.extent.depth, info.mip_levels, info.flags)
        }
        "vkCreateBuffer" => {
            let info = unsafe { &*(info as *const vk::BufferCreateInfo<'_>) };
            format!(" (size {}, usage {:?})", info.size, info.usage)
        }
        "vkAllocateMemory" => {
            let info = unsafe { &*(info as *const vk::MemoryAllocateInfo<'_>) };
            format!(" (size {}, memory type {})", info.allocation_size, info.memory_type_index)
        }
        "vkCreateImageView" => {
            let info = unsafe { &*(info as *const vk::ImageViewCreateInfo<'_>) };
            format!(" (format {:?}, type {:?})", info.format, info.view_type)
        }
        "vkCreateShaderModule" => {
            let info = unsafe { &*(info as *const vk::ShaderModuleCreateInfo<'_>) };
            let words = if info.p_code.is_null() { Vec::new() } else { (0..5usize.min(info.code_size / 4)).map(|k| unsafe { *info.p_code.add(k) }).collect::<Vec<u32>>() };
            let head: Vec<String> = words.iter().map(|w| format!("{w:08x}")).collect();
            format!(" (code size {} bytes, flags {:?}, first words {})", info.code_size, info.flags, head.join(" "))
        }
        "vkCreateDescriptorPool" => {
            let info = unsafe { &*(info as *const vk::DescriptorPoolCreateInfo<'_>) };
            format!(" (max sets {}, flags {:?})", info.max_sets, info.flags)
        }
        _ => String::new(),
    }
}

static CREATE_COMPUTE: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn trace_create_compute(
    device: vk::Device,
    cache: vk::PipelineCache,
    count: u32,
    infos: *const vk::ComputePipelineCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Pipeline,
) -> vk::Result {
    let real = unsafe { std::mem::transmute::<usize, vk::PFN_vkCreateComputePipelines>(CREATE_COMPUTE.load(Ordering::Relaxed)) };
    let result = unsafe { real(device, cache, count, infos, allocator, out) };
    if result != vk::Result::SUCCESS {
        note(format!("vkCreateComputePipelines returned {result:?} ({count} pipelines)"));
    }
    result
}

static ALLOCATE_SETS: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn trace_allocate_sets(device: vk::Device, info: *const vk::DescriptorSetAllocateInfo<'_>, out: *mut vk::DescriptorSet) -> vk::Result {
    let real = unsafe { std::mem::transmute::<usize, vk::PFN_vkAllocateDescriptorSets>(ALLOCATE_SETS.load(Ordering::Relaxed)) };
    let result = unsafe { real(device, info, out) };
    if result != vk::Result::SUCCESS {
        let count = if info.is_null() { 0 } else { unsafe { (*info).descriptor_set_count } };
        note(format!("vkAllocateDescriptorSets returned {result:?} ({count} sets)"));
    }
    result
}

static BIND_IMAGE: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn trace_bind_image(device: vk::Device, image: vk::Image, memory: vk::DeviceMemory, offset: vk::DeviceSize) -> vk::Result {
    let real = unsafe { std::mem::transmute::<usize, vk::PFN_vkBindImageMemory>(BIND_IMAGE.load(Ordering::Relaxed)) };
    let result = unsafe { real(device, image, memory, offset) };
    if result != vk::Result::SUCCESS {
        note(format!("vkBindImageMemory returned {result:?}"));
    }
    result
}

static BIND_BUFFER: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn trace_bind_buffer(device: vk::Device, buffer: vk::Buffer, memory: vk::DeviceMemory, offset: vk::DeviceSize) -> vk::Result {
    let real = unsafe { std::mem::transmute::<usize, vk::PFN_vkBindBufferMemory>(BIND_BUFFER.load(Ordering::Relaxed)) };
    let result = unsafe { real(device, buffer, memory, offset) };
    if result != vk::Result::SUCCESS {
        note(format!("vkBindBufferMemory returned {result:?}"));
    }
    result
}

static MAP_MEMORY: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn trace_map_memory(device: vk::Device, memory: vk::DeviceMemory, offset: vk::DeviceSize, size: vk::DeviceSize, flags: vk::MemoryMapFlags, out: *mut *mut c_void) -> vk::Result {
    let real = unsafe { std::mem::transmute::<usize, vk::PFN_vkMapMemory>(MAP_MEMORY.load(Ordering::Relaxed)) };
    let result = unsafe { real(device, memory, offset, size, flags, out) };
    if result != vk::Result::SUCCESS {
        note(format!("vkMapMemory returned {result:?} (size {size})"));
    }
    result
}

unsafe extern "system" fn proc_addr(device: vk::Device, name: *const c_char) -> vk::PFN_vkVoidFunction {
    let real = unsafe { std::mem::transmute::<usize, vk::PFN_vkGetDeviceProcAddr>(REAL_PROC_ADDR.load(Ordering::Relaxed)) };
    let text = if name.is_null() { &[][..] } else { unsafe { CStr::from_ptr(name) }.to_bytes() };
    if DEBUG_UTILS.contains(&text) {
        return None;
    }
    let mut function = unsafe { real(device, name) };
    if function.is_none()
        && let Some(core) = text.strip_suffix(b"KHR")
        && let Ok(core) = std::ffi::CString::new(core)
    {
        function = unsafe { real(device, core.as_ptr()) };
    }
    let Some(function) = function else {
        absent(format!("the device has no {}", String::from_utf8_lossy(text)));
        return None;
    };
    let address = function as *const () as usize;
    let replaced = create_trampoline(text, address).or_else(|| {
        let (slot, trampoline): (&AtomicUsize, usize) = match text {
            b"vkCreateComputePipelines" => (&CREATE_COMPUTE, trace_create_compute as *const () as usize),
            b"vkAllocateDescriptorSets" => (&ALLOCATE_SETS, trace_allocate_sets as *const () as usize),
            b"vkBindImageMemory" => (&BIND_IMAGE, trace_bind_image as *const () as usize),
            b"vkBindBufferMemory" => (&BIND_BUFFER, trace_bind_buffer as *const () as usize),
            b"vkMapMemory" => (&MAP_MEMORY, trace_map_memory as *const () as usize),
            _ => return None,
        };
        slot.store(address, Ordering::Relaxed);
        Some(trampoline)
    });
    match replaced {
        Some(trampoline) => Some(unsafe { std::mem::transmute::<usize, unsafe extern "system" fn()>(trampoline) }),
        None => Some(function),
    }
}
