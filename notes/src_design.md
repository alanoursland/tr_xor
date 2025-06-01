# Source Code Design for Prototype Surface Learning Experiments

## Overview

This document outlines the source code structure. The design emphasizes modularity, reproducibility, and extensibility for future experiments on higher-dimensional problems.

## File Structure

### `models.py` - Neural Network Definitions

Contains all neural network model architectures used in experiments, with specialized focus on geometric interpretability.

**Core Classes:**

**MLP Class**: Primary multi-layer perceptron implementation
- Constructor: __init__(self, input_dim, hidden_dims, output_dim, activation='relu', initialization='kaiming')
- Parameters: input_dim (int), hidden_dims (list of ints), output_dim (int), activation (str from ['relu', 'abs', 'sigmoid', 'tanh', 'gelu', 'swish']), initialization (str from ['xavier', 'kaiming', 'custom'])
- Methods: forward(x), get_hyperplanes(), get_weights_and_biases(), normalize_weights(), detect_mirror_pairs()
- Properties: num_layers, activation_type, parameter_count

**Specialized XOR Models**: Pre-configured architectures for XOR experiments
- XORNet: Generic configurable XOR solver with variable hidden units
- MinimalXORNet: Theoretical minimum architectures (single absolute value unit)
- SymmetricXORNet: Architecture designed to learn symmetric weight patterns

**Activation Function Registry**: Centralized activation function management
- register_activation(name, function, derivative): Add custom activations
- get_activation(name): Retrieve activation function by name
- list_activations(): Return available activation functions
- activation_properties(name): Return metadata (monotonic, order, separation_order)

**Weight Initialization Strategies**:
- kaiming_init(layer, nonlinearity): He initialization for ReLU-family
- xavier_init(layer, gain): Glorot initialization for sigmoid-family  
- custom_init(layer, strategy, **kwargs): Extensible custom initialization
- zero_bias_init(layer): Initialize biases to zero for centered boundaries
- prototype_aware_init(layer, prototype_points): Initialize to pass through specific points

**Geometric Analysis Methods**:
- extract_hyperplane_equations(model): Return list of (W, b) tuples
- compute_prototype_regions(model, bounds): Identify zero-activation regions
- calculate_decision_boundaries(model, resolution): Sample decision boundary points
- analyze_weight_symmetry(model): Detect mirrored weight patterns

### `data.py` - Dataset Generation and Management

Handles all data-related operations with emphasis on geometric properties and visualization support.

**Primary Dataset Functions**:

**XOR Data Generation**:
- generate_xor_data(normalized=True, center_origin=True): Standard XOR truth table
- generate_xor_variants(scales, rotations, translations): Transformed XOR datasets
- create_xor_with_noise(noise_std, num_samples): Noisy XOR for robustness testing

**Parity Problem Generation**:
- generate_parity_data(n_bits, signed=True): Higher-dimensional parity problems
- create_boolean_hypercube(n_dims): All 2^n binary combinations
- sample_parity_subset(n_bits, num_samples): Random subset of parity space

**Visualization Support Data**:
- sample_input_space(bounds, resolution, grid_type='uniform'): Dense grid for contour plots
- generate_decision_boundary_samples(model, bounds, num_points): Points near boundaries
- create_prototype_region_samples(model, region_id, density): Points within specific regions

**Data Preprocessing Utilities**:
- normalize_dataset(x, method='standardize'): Standardization and min-max scaling
- center_at_origin(x): Translate data to center at origin
- apply_rotation(x, angle): Rotate 2D data by specified angle
- add_gaussian_noise(x, std): Add noise for robustness experiments

**Batch Management**:
- create_training_batches(x, y, batch_size, shuffle=True): Standard batch creation
- create_full_batch_iterator(datasets): For small problems like XOR
- balance_classes(x, y): Ensure equal class representation

**Dataset Validation**:
- verify_xor_labels(x, y): Confirm XOR truth table correctness
- check_parity_consistency(x, y, n_bits): Validate parity problem setup
- analyze_class_separability(x, y): Linear separability analysis

### `configs.py` - Experiment Configuration

Comprehensive experiment specification system with validation and extensibility.

**Main Configuration Structure**:

**EXPERIMENTS Dictionary**: Central registry of all experiment configurations
- Each experiment key maps to complete specification dictionary
- Hierarchical organization: base configs, variants, parameter sweeps
- Inheritance support: experiments can extend base configurations

**Configuration Schema**:

**Model Configuration**:
- architecture: Dictionary specifying layer dimensions and connections
- activation: String or dictionary for per-layer activations
- initialization: Weight and bias initialization strategies
- regularization: Dropout, weight decay, and other regularization parameters

**Training Configuration**:
- optimizer: Type (Adam, SGD, RMSprop) with hyperparameters
- learning_rate: Fixed value or schedule specification
- epochs: Maximum training iterations
- convergence_criteria: Early stopping conditions based on loss or accuracy
- batch_size: Training batch configuration
- loss_function: Cross-entropy, MSE, or custom loss specifications

**Data Configuration**:
- problem_type: XOR, parity, or custom dataset specification
- normalization: Standardization, centering, scaling options
- augmentation: Noise, rotation, translation parameters
- validation_split: Fraction for validation (if applicable)

**Analysis Configuration**:
- geometric_analysis: Enable hyperplane plotting, region identification
- weight_analysis: Track weight evolution, detect patterns
- activation_analysis: Monitor activation patterns across inputs
- convergence_analysis: Track training dynamics and final states
- visualization_options: Plot types, resolution, styling preferences

**Execution Configuration**:
- num_runs: Number of independent training runs
- random_seeds: Specific seeds for reproducibility
- device: CPU/GPU specification
- logging_level: Verbosity of output
- save_intermediate: Whether to save models during training

**Pre-defined Experiment Templates**:
- xor_relu_basic: Standard 2-hidden-unit ReLU network
- xor_abs_minimal: Single absolute value unit
- xor_sigmoid_classic: Traditional sigmoid approach
- parity_3bit_relu: Three-bit parity with ReLU
- activation_comparison: Systematic comparison across activations

**Configuration Validation Functions**:
- validate_experiment_config(config): Check completeness and consistency
- resolve_config_inheritance(config_name): Handle config extension
- generate_parameter_sweep(base_config, sweep_params): Create grid search configs
- merge_configs(base, override): Combine configuration dictionaries

### `run.py` - Experiment Execution

Central orchestration system for running experiments with comprehensive logging and state management.

**Command Line Interface**:

**Primary Entry Points**:
- Main execution function: run_experiment(experiment_name, num_runs, device, verbose)
- Batch execution: run_experiment_batch(experiment_list, parallel)
- Parameter sweep: run_parameter_sweep(base_experiment, param_grid)

**Command Line Arguments**:
- --experiment: Experiment name from configs.py
- --runs: Number of independent runs (default from config)
- --device: Force CPU/GPU usage
- --verbose: Logging verbosity level
- --output-dir: Override default results directory
- --resume: Resume interrupted experiment runs
- --dry-run: Validate configuration without execution

**Execution Pipeline**:

**Initialization Phase**:
- load_and_validate_config(experiment_name): Load and validate experiment specification
- setup_output_directories(base_dir, experiment_name): Create structured result directories
- initialize_logging(output_dir, verbosity): Configure logging to file and console
- set_random_seeds(seeds_list): Ensure reproducible execution across runs

**Training Loop Management**:
- create_model_from_config(model_config): Instantiate model with specified architecture
- setup_training_components(training_config): Initialize optimizer, loss function, scheduler
- execute_training_run(model, data, training_config): Single training execution
- monitor_training_progress(model, losses, config): Track convergence and save checkpoints

**Multi-Run Coordination**:
- coordinate_multiple_runs(experiment_config, num_runs): Manage independent training runs
- aggregate_run_results(run_results): Combine statistics across runs
- detect_convergence_failures(run_results): Identify failed or divergent training
- save_experiment_summary(aggregated_results, output_dir): Create summary statistics

**State Management**:
- save_run_state(run_id, model, optimizer, epoch, losses): Checkpoint individual runs
- load_run_state(checkpoint_path): Resume interrupted training
- save_final_model(model, config, stats, output_path): Save completed model with metadata
- create_experiment_manifest(experiment_config, run_results): Document experiment execution

**Progress Monitoring**:
- real_time_progress_display(current_run, total_runs, epoch, loss): Live training updates
- estimate_completion_time(start_time, current_progress, total_work): Time estimates
- log_training_milestones(epoch, loss, accuracy, learning_rate): Structured logging
- generate_training_summaries(losses, accuracies, config): Create training reports

**Output Organization**:
Results directory structure:
- experiment_name/run_001/model.pth, config.json, training_log.json, analysis/
- experiment_name/run_002/...
- experiment_name/summary/aggregated_stats.json, best_models/, comparative_analysis/

### `analyze.py` - Post-Experiment Analysis and Visualization

Analysis system with rich visualization capabilities.

**Geometric Analysis Functions**:

**Hyperplane Analysis**:
- extract_learned_hyperplanes(model): Extract (W, b) parameters from all layers
- plot_hyperplanes_2d(hyperplanes, data_points, bounds, style_config): 2D hyperplane visualization
- visualize_prototype_regions(model, input_bounds, resolution): Map zero-activation regions
- compute_hyperplane_intersections(hyperplanes): Find intersection points and lines
- analyze_hyperplane_orientations(hyperplanes): Study normal vector patterns

**Distance Field Analysis**:
- compute_distance_fields(model, input_space, distance_type): Calculate distance to learned surfaces
- plot_distance_contours(distance_field, model, data_points): Contour plots of distance functions
- analyze_prototype_proximity(data_points, model): Measure proximity of data to prototype surfaces
- visualize_activation_landscapes(model, input_bounds): 3D visualization of activation functions

**Prototype Region Analysis**:
- identify_prototype_regions(model, input_bounds): Map regions where activations are zero
- compute_region_membership(data_points, prototype_regions): Classify points by region membership
- analyze_region_coverage(prototype_regions, data_distribution): Coverage statistics
- measure_region_stability(models_list, prototype_regions): Consistency across training runs

**Weight Pattern Analysis**:
- detect_mirror_weight_pairs(model, similarity_threshold): Find W_i ≈ -W_j patterns
- analyze_weight_symmetries(model): Detect rotational and reflectional symmetries
- track_weight_evolution(training_logs, analysis_epochs): Monitor weight changes during training
- compute_weight_clustering(models_list): Cluster analysis of learned weight patterns

**Activation Pattern Analysis**:
- compute_activation_signatures(model, input_set): Activation patterns for each input
- analyze_zero_activation_patterns(model, data_points): Map zero-output neuron patterns
- measure_activation_sparsity(model, input_distribution): Sparsity statistics
- visualize_activation_heatmaps(activation_patterns, input_labels): Heat map visualizations

**Comparative Analysis**:

**Cross-Activation Function Comparison**:
- compare_learned_geometries(models_dict, comparison_metrics): Compare across activation types
- analyze_convergence_patterns(training_logs_dict): Compare training dynamics
- measure_solution_consistency(models_dict, num_runs): Stability across activation functions
- generate_activation_comparison_report(analysis_results): Structured comparison document

**Multi-Run Statistical Analysis**:
- aggregate_geometric_statistics(run_results_list): Statistics across multiple runs
- analyze_solution_diversity(models_list, diversity_metrics): Measure solution variation
- detect_consistent_patterns(analysis_results_list): Find reproducible patterns
- generate_statistical_summaries(aggregated_data): Summary statistics and confidence intervals

**Visualization System**:

**Plot Generation Functions**:
- create_hyperplane_plots(model, data, style_config, output_path): Standard hyperplane visualizations
- generate_prototype_region_plots(model, regions, data, output_path): Region boundary plots
- create_activation_landscape_plots(model, bounds, resolution, output_path): 3D activation landscapes
- plot_weight_evolution_timeseries(weight_history, output_path): Weight change over time

**Styling and Configuration**:
- load_visualization_style(style_name): Load predefined visual styles
- configure_plot_aesthetics(style_config): Set colors, fonts, sizes
- generate_publication_quality_plots(analysis_results, style): High-quality output
- create_interactive_visualizations(model, data, output_path): Interactive plots using plotly

**Report Generation**:
- generate_analysis_report(experiment_results, template): Comprehensive analysis document
- create_executive_summary(key_findings, metrics): High-level summary
- generate_technical_appendix(detailed_analysis, data): Detailed technical results
- export_analysis_data(analysis_results, format): Export to CSV, JSON, or HDF5

**Command Line Interface**:
- --experiment: Specify experiment to analyze
- --visualization: Enable/disable specific visualization types
- --metrics: Select which metrics to compute
- --output-format: Choose output formats (PNG, PDF, SVG)
- --style: Select visualization style theme
- --interactive: Generate interactive visualizations

### `utils.py` - General Utility Functions

**File I/O Operations**:

**Model Management**:
- save_model_with_metadata(model, metadata, filepath): Save model with comprehensive metadata
- load_model_with_validation(filepath, expected_architecture): Load and validate model architecture
- batch_save_models(models_dict, base_path): Save multiple models with organized naming
- create_model_checkpoints(model, optimizer, epoch, checkpoint_dir): Training checkpoints
- archive_experiment_results(experiment_dir, archive_path): Compress completed experiments

**Configuration Management**:
- save_config_with_timestamp(config, filepath): Save configuration with execution metadata
- load_and_validate_config(filepath, schema): Load with schema validation
- merge_configuration_files(base_config_path, override_config_path): Combine configurations
- generate_config_hash(config): Create unique hash for configuration identification
- export_config_to_formats(config, output_dir): Export to YAML, JSON, TOML

**Results Serialization**:
- serialize_analysis_results(results, filepath, format): Support multiple output formats
- deserialize_analysis_results(filepath): Load with format auto-detection
- create_results_database(results_list, database_path): SQLite database for queryable results
- export_results_to_csv(results, filepath, flatten): Flatten nested results for spreadsheet analysis

**Logging System**:

**Logger Configuration**:
- setup_experiment_logging(output_dir, experiment_name, verbosity): Configure structured logging
- create_multi_level_logger(name, handlers_config): Support file, console, and remote logging
- configure_training_logger(model_name, output_path): Specialized training progress logging
- setup_analysis_logging(analysis_type, output_dir): Analysis-specific logging configuration

**Logging Utilities**:
- log_training_progress(epoch, loss, accuracy, learning_rate, logger): Structured training logs
- log_hyperparameter_config(config, logger): Log all hyperparameters with proper formatting
- log_model_architecture(model, logger): Detailed architecture logging
- log_analysis_results(results, logger): Structure analysis result logging
- create_training_timeline(training_logs): Generate timeline visualization of training

**Random Seed Management**:

**Reproducibility Functions**:
- set_global_random_seeds(seed): Set seeds for torch, numpy, random, and Python hash
- generate_experiment_seeds(base_seed, num_runs): Generate reproducible seed sequences
- save_seed_state(filepath): Save current random state for exact reproduction
- restore_seed_state(filepath): Restore exact random state
- validate_reproducibility(model_factory, data, seed, num_trials): Test reproducibility

**Mathematical Utilities**:

**Geometric Computations**:
- compute_point_to_hyperplane_distance(point, weights, bias): Exact geometric distance
- find_hyperplane_intersections(hyperplane_list): Compute intersection points and lines
- compute_hyperplane_angles(weights_1, weights_2): Angle between normal vectors
- project_point_onto_hyperplane(point, weights, bias): Orthogonal projection
- compute_convex_hull_of_regions(region_points): Convex hull of prototype regions

**Distance and Similarity Metrics**:
- cosine_similarity_matrix(weight_matrix): Pairwise cosine similarities
- euclidean_distance_matrix(points): Pairwise Euclidean distances  
- compute_weight_clustering_metrics(weight_history): Cluster analysis metrics
- measure_geometric_stability(hyperplanes_list): Stability across training runs

**Statistical Analysis**:
- compute_convergence_statistics(loss_curves): Convergence rate and stability metrics
- analyze_weight_distribution(weights, distribution_type): Distribution fitting and analysis
- perform_hyperparameter_sensitivity_analysis(results_grid): Sensitivity to hyperparameters
- calculate_confidence_intervals(sample_data, confidence_level): Bootstrap confidence intervals

**Training Utilities**:

**Training Support Functions**:
- create_loss_function(loss_type, **kwargs): Factory for loss function creation
- setup_optimizer(optimizer_type, model_parameters, **kwargs): Optimizer factory with configuration
- create_learning_rate_scheduler(scheduler_type, optimizer, **kwargs): LR scheduler factory
- implement_early_stopping(patience, min_delta, restore_best): Early stopping callback

**Convergence Analysis**:
- detect_training_convergence(loss_history, criteria): Automatic convergence detection
- analyze_training_stability(loss_curves, window_size): Training stability metrics
- identify_training_phases(loss_history, gradient_threshold): Identify distinct training phases
- compute_training_efficiency_metrics(training_logs): Efficiency and speed metrics

**Model Analysis Utilities**:
- count_model_parameters(model): Count total and trainable parameters
- analyze_gradient_flow(model, input_batch): Gradient flow analysis
- compute_model_complexity_metrics(model): Various complexity measures
- generate_model_summary_report(model, input_shape): Comprehensive model description

**Directory and Path Management**:
- create_experiment_directory_structure(base_path, experiment_name): Standard directory creation
- generate_unique_experiment_id(experiment_name, timestamp): Unique experiment identification
- organize_results_by_date(results_dir): Organize results chronologically
- cleanup_incomplete_experiments(base_dir, min_age_hours): Clean up failed experiment directories
- archive_old_experiments(base_dir, archive_dir, age_threshold): Archive completed experiments

## Design Principles

### Modularity
Each file has a clearly defined responsibility with minimal coupling between components. The interfaces between modules are well-defined and stable, allowing independent development and testing of each component.

### Extensibility
The architecture easily accommodates future extensions including new activation functions (GELU, Swish, custom functions), different network architectures, higher-dimensional problems beyond XOR, new analysis metrics and visualizations, and integration with external tools and libraries.

### Reproducibility
Complete reproducibility is ensured through consistent random seed handling across all components, comprehensive configuration logging with version tracking, deterministic experiment execution with checkpoint support, and complete audit trails of all experimental procedures.

### Prototype Surface Theory Focus
The entire pipeline is specifically designed to facilitate investigation of prototype surface learning through specialized geometric interpretation tools, surface-based analysis rather than traditional decision boundary analysis, distance-based metrics, and visualization systems optimized around prototype surface theory predictions.

## Implementation Priorities

### Phase 1: Core Infrastructure
Implement basic versions of models.py, data.py, and utils.py to establish the foundation for experimentation with XOR problems.

### Phase 2: Experiment Framework
Complete configs.py and run.py to enable systematic experimentation with proper logging and state management.

### Phase 3: Analysis System
Implement analyze.py with comprehensive geometric analysis and visualization capabilities specifically focused on prototype surface theory validation.

### Phase 4: Advanced Features
Add interactive visualizations, real-time training monitoring, automated report generation, and integration with external analysis tools.

This design provides a robust foundation for investigating the Prototype Surface Learning theory while maintaining the flexibility to evolve as new insights and requirements emerge from the research process.