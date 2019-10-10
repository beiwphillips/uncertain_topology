# uncertain_topology

A repository for housing an uncertain topology computation algorithm

# Installation (Mac)

Install swig:

```
    brew install swig
```

I recommend creating a virtual environment using conda and activating it:

```
    conda env create --name utpy
    conda activate utpy (possibly source activate utpy)
```

Checkout and build flatpy:

```
    git clone https://github.com/maljovec/flatpy.git
    cd flatpy
    python setup.py develop
```

Checkout and build a compatible version of nglpy:

```
    git clone https://github.com/maljovec/nglpy.git
    cd nglpy
    git checkout new_api
    make
    CFLAGS=-stdlib=libc++ python setup.py develop
```

You may have to copy the .so file into nglpy (you could possibly get by with a symlink, but that didn't work for me):

```
    cp _ngl*.so nglpy/
```

Checkout and build the compatible version of topopy:

```
    git clone https://github.com/maljovec/topopy.git
    cd topopy
    git checkout uncertain
    make
    CFLAGS=-stdlib=libc++ python setup.py develop
```

Again, you may have to copy the .so file into topopy (you could possibly get by with a symlink, but that didn't work for me):

```
    cp _topology*.so topopy/
```

Checkout and build the uncertain topology library:

```
    git clone https://dmaljovec@bitbucket.org/dmaljovec/uncertain_topology.git
```

Install other prerequisite libraries that are not handled above:

```
    pip install matplotlib seaborn scikit-image
```

Now you should be good to go!

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


