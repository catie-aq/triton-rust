#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_void, c_char, c_int};
use std::mem;
use std::slice;

use ndarray::{Array, Array3, Array4};

include!(concat!(env!("OUT_DIR"), "/shared_memory_binding.rs"));

pub struct SystemSharedMemoryRegionHandle {
    name: String,
    key: String,
    handle: *mut c_void
}

impl SystemSharedMemoryRegionHandle {
    pub fn create(triton_shm_name: &'static str, shm_key: &'static str, size: u64) -> Self {

        let c_triton_shm_name = CString::new(triton_shm_name).unwrap();
        let c_shm_key = CString::new(shm_key).unwrap();
        let mut handle: *mut c_void = std::ptr::null_mut();

        let result = unsafe {
            SharedMemoryRegionCreate(
                c_triton_shm_name.as_ptr(),
                c_shm_key.as_ptr(),
                size,
                &mut handle
            )
        };

        SystemSharedMemoryRegionHandle {
            name: triton_shm_name.to_string(),
            key: shm_key.to_string(),
            handle: handle,
        }
    }

    pub fn get_name(&mut self) -> String {
        self.name.clone()
    }

    pub fn destroy(&mut self) {
        let result = unsafe {
            SharedMemoryRegionDestroy(
                self.handle
            )
        };
    }

    pub fn copy_array(&mut self, array: &Array4<f32>) {

        let byte_size = array.shape().iter().product::<usize>() * mem::size_of::<f32>();

        let result = unsafe { SharedMemoryRegionSet(
                self.handle,
                0,
                byte_size as u64,
                array.as_ptr() as *const c_void
            )
        };
    }

    pub fn get_data(&mut self, size: u64, offset: u64) -> Vec<u8> {

        let mut shm_addr: *mut c_char = std::ptr::null_mut();
        let mut shm_key: *const c_char = unsafe { std::mem::uninitialized() };
        let mut fd: c_int = 0;
        let mut size_val: size_t = 0;
        let mut offset_val: size_t = 0;

        let result = unsafe { GetSharedMemoryHandleInfo(
                self.handle,
                &mut shm_addr,
                &mut shm_key,
                &mut fd,
                &mut size_val,
                &mut offset_val
            )
        };

        let result = unsafe {
            slice::from_raw_parts(shm_addr.add(offset.try_into().unwrap()) as *mut u8, size.try_into().unwrap())
        };

        result.to_vec()
    }
}
