# uncertain_topology

A repository for housing an uncertain topology computation algorithm

# Installation

Begin by checking out this library:

```
    git clone https://bitbucket.org/dmaljovec/uncertain_topology.git
```

## Setting up a Virtual Environment

We recommend creating a virtual environment and activating it:

```
    pip install virtualenv
    virtualenv utpy
    source utpy/bin/activate
```

Install prerequisites:

```
    pip install -f requirements.txt
```

Now you should be good to go!

## Using the Pre-configured Docker Container

Another method for using this library is to use our pre-built docker container.


# Available Notebooks:

* Ackley analysis:
 - Ackley\ Analysis.ipynb
  ```
  uniform_synthetics.py -f ackley -c 50
  ```

* Himmelblau analysis:
 - Himmelblau\ Analysis.ipynb
  ```
  uniform_synthetics.py -f himmelblau -c 50
  ```

* Karman Flow:
  - Flow Overlays.ipynb
  ```
  python real_world_analysis.py  -f upSampled -n 17
  ```

* Bei Flow:
  - Bei Flow.ipynb
  ```
  python real_world_analysis.py -f flowDataFromBei -n 10
  ```

# Other Notebook explanations:

* Get Forecast Ensemble.ipynb - A notebook for pulling weather forecast data from a website
* Persistence Plot.ipynb - A notebook for generating one of the other figures in the paper demonstrating persistence simplification
* Uncertain Morse Complex Figure Generator.ipynb - I am not sure that this has anything useful
* UTPy Examples.ipynb - Same, I am not sure this one has anything of value


