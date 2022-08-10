#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{CString};
use std::os::raw::{c_void, c_char};

include!(concat!(env!("OUT_DIR"), "/shared_memory_binding.rs"));

pub struct CudaSharedMemoryRegionHandle {
    name: String,
    handle: *mut c_void
}

impl CudaSharedMemoryRegionHandle {
    pub fn create(triton_shm_name: &'static str, size: u64, device_id: i64) -> Self {

        let c_triton_shm_name = CString::new(triton_shm_name).unwrap();
        let mut handle: *mut c_void = std::ptr::null_mut();

        let _result = unsafe {
            CudaSharedMemoryRegionCreate(
                c_triton_shm_name.as_ptr(),
                size,
                device_id.try_into().unwrap(),
                &mut handle
            )
        };

        CudaSharedMemoryRegionHandle {
            name: triton_shm_name.to_string(),
            handle: handle,
        }
    }

    pub fn from_ptr(triton_shm_name: &'static str, ptr: *mut c_void) -> Self {

        CudaSharedMemoryRegionHandle {
            name: triton_shm_name.to_string(),
            handle: ptr,
        }
    }

    pub fn get_name(&mut self) -> String {
        self.name.clone()
    }

    pub fn get_raw_handle(&mut self) -> Vec<u8> {

        let mut raw_handle_ptr: *mut c_char = std::ptr::null_mut();

        let _result = unsafe {
            CudaSharedMemoryGetRawHandle(
                self.handle,
                &mut raw_handle_ptr
            )
        };

        let raw_handle_str = unsafe { CString::from_raw(raw_handle_ptr) };
        println!("{:?}", raw_handle_str);
        let raw_handle = raw_handle_str.into_bytes();
        println!("{:?}", raw_handle);
        println!("{:?}", raw_handle.len());

        raw_handle
    }

    pub fn destroy(&mut self) {

        let _result = unsafe {
            CudaSharedMemoryRegionDestroy(
                self.handle
            )
        };
    }
}
