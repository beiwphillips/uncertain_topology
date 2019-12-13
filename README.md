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

Another method for using this library is to use our pre-built docker container:

```
docker build -t utpy .
docker run -it utpy bash
```

This will drop you into a terminal session of the running container.

# Analysis provided

## Ackley analysis:
 Ackley\ Analysis.ipynb - Notebook generating some of the figures seen in the paper regarding the Ackley synthetic function, the remainder are generated with the following script:
 
  ```
  uniform_synthetics.py -f ackley -c 50
  ```

## Himmelblau analysis:
 Himmelblau\ Analysis.ipynb - Notebook generating many figures seen in the paper regarding the Himmelblau synthetic function, the remainder are generated with the following script:
  ```
  uniform_synthetics.py -f himmelblau -c 50
  ```

## Karman Flow:
  Flow Overlays.ipynb - Notebook generating many figures seen in the paper regarding the Karman flow dataset, the remainder are generated with the following script:
  ```
  python real_world_analysis.py  -f upSampled -n 17
  ```

## Bei Flow:
  Bei Flow.ipynb - Notebook generating many figures seen in the paper regarding the secondary flow dataset, the remainder are generated with the following script:
  ```
  python real_world_analysis.py -f flowDataFromBei -n 10
  ```

## Other Notebook explanations:

* Get Forecast Ensemble.ipynb - A notebook for pulling weather forecast data from a website
* Persistence Plot.ipynb - A notebook for generating one of the other figures in the paper demonstrating persistence simplification
* Uncertain Morse Complex Figure Generator.ipynb - I am not sure that this has anything useful
* UTPy Examples.ipynb - Same, I am not sure this one has anything of value


# Disclaimer

This code is in a pre-alpha state and is provided as-is:

![Certified Works on my Machine](https://blog.codinghorror.com/content/images/uploads/2007/03/6a0120a85dcdae970b0128776ff992970c-pi.png)

This code is more a proof-of-concept rather than production-ready software. There is an underlying package `utpy` that attempts to bundle up some of the concepts into a reusable library, but even this lacks documentation, testing, and does not adhere to best practices. I am happy to answer quetions and work with someone who wants to maintain this project, but I do not have the time or ambition to lead this project. Happy Coding!
