/* Copyright CATIE, 2022-2023

b.albar@catie.fr

This software is governed by the CeCILL-B license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-B
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-B license and that you accept its terms.*/

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
    pub fn create(triton_shm_name: &'static str, size: usize, device_id: i64) -> Self {

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
