# Helpers Functions Details

Here is a simplified explanation of each function in `helpers.py`:

1. **`scale_data`**: Adjusts your data so it has a standard range, making it easier for the AI to learn. It also returns the rules (mean and standard deviation) used for this adjustment.
2. **`unscale_data`**: Reverses the adjustment made by `scale_data`, converting the AI's scaled predictions back into normal, real-world numbers.
3. **`scale_data_with_scaler`**: Applies previously calculated adjustment rules to new data.
4. **`save_scaler_params`**: Saves the adjustment rules to a file so they can be reused later.
5. **`load_scaler_params`**: Loads previously saved adjustment rules from a file.
6. **`get_train_val_dataloaders`**: Splits your data into two groups (one for training the AI, one for checking its progress) and prepares them to be fed into the AI in small batches.
7. **`get_standard_trainer`**: Sets up the main engine that trains the AI, configuring things like when to stop early if it's not improving and where to save its progress.
8. **`get_latest_log_and_checkpoint_path`**: Finds the folder containing the most recent training run and the latest saved state of the AI model.
9. **`find_latest_checkpoint`**: Looks inside a specific folder and finds the most recently saved AI model file.
10. **`get_latest_version_dir`**: Finds the folder for the most recent training session.
11. **`create_next_version_dir`**: Creates a new, sequentially numbered folder for a new training session so old ones aren't overwritten.
12. **`visualize_training_logs`**: Creates a graph showing how the AI's learning errors decreased over time during training.
13. **`load_narx_dataset_with_metadata`**: Opens and reads the main dataset file, separating the actual data numbers from information about how the data is structured (metadata).
14. **`create_feature_structure_from_metadata`**: Reads the dataset information to understand how the past system states are organized.
15. **`create_input_feature_structure_from_metadata`**: Reads the dataset information to understand how the control inputs are organized.
16. **`create_label_structure_from_metadata`**: Reads the dataset information to understand what exactly the AI is supposed to predict.
17. **`save_model_metadata`**: Gathers all important configuration details about the AI and its data, and saves them to a file.
18. **`load_model_metadata`**: Reads the saved configuration file to understand an existing AI model's setup.
19. **`load_narx_model`**: Fully loads a pre-trained AI model along with its scaling rules and configuration, making it ready to make predictions.
20. **`get_train_val_test_dataloaders`**: Splits the data into three groups: training, validation (checking progress), and testing (a final exam for the AI), and readies them for processing.
21. **`evaluate_on_test_set`**: Tests a fully trained AI model on the "final exam" data and calculates scores to show how well it performs.
22. **`recursive_narx_rollout`**: Makes the AI predict multiple steps into the future by using its own past predictions as the inputs for its next prediction.
