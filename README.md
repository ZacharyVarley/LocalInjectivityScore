# Local Injectivity Score

Repository for analyzing local injectivity from a control space X to an outcome
space Y via a Leave one out (LOO) residual for a local linearization of the
inverse mapping. Includes basic Cahn-Hilliard and Potts model data generation.

## Project Structure

- **ch_ab_*** - Cahn-Hilliard (alpha / beta control space) analysis scripts and data
  - `ch_ab_gen.py` - Data generation for baker map simulations
  - `ch_ab_analyze_explained_fraction.py` - Analysis of explained variance
  - `ch_ab_publication_figures.py` - Publication-ready figure generation
  - `ch_ab_visualize_control_space.py` - Control space visualization

- **ch_tprofiles_*** - Temperature profile version of Cahn-Hilliard simulations
  - `ch_tprofiles_gen.py` - Data generation for temperature profile simulations
  - `ch_tprofiles_analyze_explained_fraction.py` - Analysis scripts
  - `ch_tprofiles_figures.py` - Figure generation

- **potts_*** - Potts model analysis
  - `potts_gen.py` - Potts model data generation
  - `potts_analyze_explained_fraction.py` - Potts data analysis

- **Utilities**
  - `injectivity_analysis_helpers.py` - Helper functions for injectivity analysis
  - `ch_plot_utils.py` - Plotting utilities
  - `intro_figure.py` - Introductory figures
  - `toy_figures.py` - Toy model visualizations