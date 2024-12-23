# csc2529_project

This is an implementation of Jos Stam's stable fluids [paper](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf). With some implementation advice taken from this follow-up [work](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)

To install dependencies, use
`pip install -r requirements.txt`.

## Implementation Details

The main operators of the stable fluids algorithm are implemented with warp kernels. We solve for the pressure and viscosity using an iterative Gauss-Sidel solver for the 2D example. 
To run the 2D example:
`python3 fluid_warp_2d.py`
