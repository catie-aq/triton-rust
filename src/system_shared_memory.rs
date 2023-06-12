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
use std::os::raw::{c_void, c_char, c_int};
use std::mem;
use std::slice;

use ndarray::{ArrayBase, RawData, Dimension};

include!(concat!(env!("OUT_DIR"), "/shared_memory_binding.rs"));

pub struct SystemSharedMemoryRegionHandle {
    name: String,
    key: String,
    handle: *mut c_void
}

impl SystemSharedMemoryRegionHandle {
    pub fn create(triton_shm_name: &'static str, shm_key: &'static str, size: usize) -> Self {

        let c_triton_shm_name = CString::new(triton_shm_name).unwrap();
        let c_shm_key = CString::new(shm_key).unwrap();
        let mut handle: *mut c_void = std::ptr::null_mut();

        let _result = unsafe {
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
        let _result = unsafe {
            SharedMemoryRegionDestroy(
                self.handle
            )
        };
    }

    pub fn copy_array<T: RawData, D: Dimension>(&mut self, array: &ArrayBase<T, D>, offset: usize) {

        let byte_size = array.shape().iter().product::<usize>() * mem::size_of::<T::Elem>();

        let _result = unsafe { SharedMemoryRegionSet(
                self.handle,
                offset,
                byte_size,
                array.as_ptr() as *const c_void
            )
        };
    }

    pub fn get_data<T: Copy>(&mut self, size: u64, offset: u64) -> Vec<T> {

        let mut shm_addr: *mut c_char = std::ptr::null_mut();
        let mut shm_key = mem::MaybeUninit::<*const c_char>::uninit();
        let mut fd: c_int = 0;
        let mut size_val: usize = 0;
        let mut offset_val: usize = 0;

        let _result = unsafe { GetSharedMemoryHandleInfo(
                self.handle,
                &mut shm_addr,
                shm_key.as_mut_ptr(),
                &mut fd,
                &mut size_val,
                &mut offset_val
            )
        };

        let result = unsafe {
            slice::from_raw_parts(shm_addr.add(offset.try_into().unwrap()) as *mut u8, size.try_into().unwrap())
        };

        let result_vec_ref = result.to_vec();
        let result_vec_ref_T = unsafe { result_vec_ref.align_to::<T>().1 };
        let result_vec: Vec<T> = result_vec_ref_T.iter().map(|x: &T| *x).collect();

        result_vec
    }
}
