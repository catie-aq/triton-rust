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

use std::process;
use ndarray::{Array};
use std::vec::Vec;
use std::collections::HashMap;

use tokenizers::tokenizer::{Result, Tokenizer};

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

fn main() -> Result<()> {
    let mut triton_inferer = TritonInference::connect("http://127.0.0.1:71").unwrap();

    /* Check if the model is running */
    let response = triton_inferer.is_model_ready("distilbert-base-uncased", "1").unwrap();
    if response == false {
        println!("{}", "Model is not running");
        process::exit(1);
    }

    let model_metadata = triton_inferer.get_model_metadata("distilbert-base-uncased", "1").unwrap();

    /* Initialize the tokenizer with the proper model */
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

    /* Encode the sentence */
    let encoding = tokenizer.encode("Hello world!", false)?;
    let tokens = encoding.get_ids().iter().map(|x| *x as i64).collect();
    let mask = encoding.get_attention_mask().iter().map(|x| *x as i64).collect();

    let tokens_array = Array::from_vec(tokens);
    let mask_array = Array::from_vec(mask);

    /* Prepare the data to be sent to Triton */
    let tokens_content = triton_inferer.get_input_content_from_ndarray(&tokens_array);
    let mask_content = triton_inferer.get_input_content_from_ndarray(&mask_array);

    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(2);
    let mut inputs_content = Vec::<Vec<u8>>::with_capacity(2);

    infer_inputs.push(triton_inferer.get_infer_input("input_ids", "INT64", &[1, 3], HashMap::new()));
    inputs_content.push(tokens_content);

    infer_inputs.push(triton_inferer.get_infer_input("attention_mask", "INT64", &[1, 3], HashMap::new()));
    inputs_content.push(mask_content);


    let response  = triton_inferer.infer("distilbert-base-uncased", "1", "25", infer_inputs, Vec::<InferRequestedOutputTensor>::new(), inputs_content).unwrap();

    /* Get the logits raw data */
    let output_logits = unsafe { response.raw_output_contents[0].align_to::<f32>().1 };
    let array_nd = Array::from_iter(output_logits.iter());
    println!("{:?}", array_nd.into_shape((1, 3, model_metadata.outputs[0].shape[2] as usize)));

    Ok(())
}
