/* Copyright CATIE, 2022

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

use std::error::Error;
use ndarray::{Array, Array3, Array4, arr3};
use std::vec::Vec;
use std::collections::HashMap;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

use triton_rust::cuda_shared_memory::CudaSharedMemoryRegionHandle;
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;

fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://0.0.0.0:7001").unwrap();

    /* Create a shared memory zone for exchanging data */

    let size_of_output = 8*768*4;

    triton_inferer.unregister_system_shared_memory("input_ids_data");
    triton_inferer.unregister_system_shared_memory("attention_mask_data");

    let mut system_mem_zone_ids = triton_inferer.create_system_shared_memory("input_ids_data", "/input_ids_data", 512).unwrap();
    let mut system_mem_zone_mask = triton_inferer.create_system_shared_memory("attention_mask_data", "/attention_mask_data", 512).unwrap();

    let mut system_mem_zone_output = triton_inferer.create_system_shared_memory("output_data", "/output_data", size_of_output).unwrap();

    /* Create an array representing some sequence of tokens */
    let ids = Array::from_shape_vec((1, 8), vec![101, 8292, 6895, 9765, 4895, 3231, 999, 102]).unwrap();
    let mask = Array::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();

    /* Infer by transfering data through the shared memory */
    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(2);
    let ids_input_params = triton_inferer.get_system_shared_memory_params("input_ids_data", 8*8, 0);
    infer_inputs.push(triton_inferer.get_infer_input("input_ids", "INT64", &[1, 8], ids_input_params));

    let mask_input_params = triton_inferer.get_system_shared_memory_params("attention_mask_data", 8*8, 0);
    infer_inputs.push(triton_inferer.get_infer_input("attention_mask", "INT64", &[1, 8], mask_input_params));

    system_mem_zone_ids.copy_array(&ids, 0);
    system_mem_zone_mask.copy_array(&mask, 0);

    let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(1);
    let output_params = triton_inferer.get_system_shared_memory_params("output_data", size_of_output, 0);
    infer_outputs.push(triton_inferer.get_infer_output("last_hidden_state", output_params));

    let response  = triton_inferer.infer("distilcamembert", "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

    /* Parse the output of the model */
    let output_class: Vec<f32> = system_mem_zone_output.get_data(size_of_output, 0);
    let array_nd = Array::from_iter(output_class.iter());
    println!("{:?}", array_nd.into_shape((1,8,768)));

    /* Unregister shared memory zone */
    let response = triton_inferer.unregister_system_shared_memory("input_ids_data").unwrap();
    let response = triton_inferer.unregister_system_shared_memory("attention_mask_data").unwrap();
    let response = triton_inferer.unregister_system_shared_memory("output_data").unwrap();

    system_mem_zone_ids.destroy();
    system_mem_zone_mask.destroy();
    system_mem_zone_output.destroy();

    Ok(())
}