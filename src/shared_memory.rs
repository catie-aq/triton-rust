#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::CString;
use std::os::raw::{c_void, c_char};
use std::mem;

include!(concat!(env!("OUT_DIR"), "/shared_memory_binding.rs"));

pub struct CudaSharedMemoryRegionHandle {
    name: String,
    handle: *mut c_void
}

impl CudaSharedMemoryRegionHandle {
    pub fn create(triton_shm_name: &'static str, size: u64, device_id: i64) -> Self {

        let c_triton_shm_name = CString::new(triton_shm_name).unwrap();
        let mut handle: *mut c_void = std::ptr::null_mut();

        let result = unsafe {
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

    pub fn get_raw_handle(&mut self) -> Vec<u8> {
        let mut raw_handle = Vec::<i8>::with_capacity(32); // check the size of the struct
        let mut buffer: *mut i8 = raw_handle.as_mut_ptr();

        let result = unsafe {
            CudaSharedMemoryGetRawHandle(
                self.handle,
                &mut buffer
            )
        };

        println!("{:?}", result);

        let converted_handle: Vec<u8> = unsafe {
            mem::transmute::<Vec<i8>, Vec<u8>>(raw_handle)
        };

        converted_handle
    }
}
