# zcf24
[![PyPI version](https://badge.fury.io/py/zcf24.svg)](https://badge.fury.io/py/zcf24)
[![Build Status](https://github.com/clemson-cal/zcf24/workflows/CI/badge.svg)](https://github.com/clemson-cal/zcf24/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This is a Python module to model AGN disk emission in changing-look inspirals.

## Installation
You can install the package using pip:

```bash
pip install zcf24
```

## Example Usage
This code below generates Fig. 1 from the paper:

```python
from matplotlib.pyplot import figure, show
from numpy import linspace
from zcf24 import DiskModel

fig = figure()
ax1 = fig.add_subplot()
disk = DiskModel(n=0.5, ell=0.85)

for t in [-8.0, -4.0, -2.0, -1.0]:
    rnu = disk.radius_of_influence(t)
    ra = linspace(disk.r_star(t), rnu, 10000)
    rb = linspace(rnu, 5.0, 10000)
    sa = [disk.surface_density(ri, t) for ri in ra]
    sb = [disk.surface_density(ri, t) for ri in rb]
    ax1.plot(ra, sa, c="purple", ls="--", lw=1.0)
    ax1.plot(
        rb,
        sb,
        c="purple",
        lw=4.0,
        ls="-",
        alpha=0.25,
    )
    if t != -2.0:
        rnu = disk.radius_of_influence(t)
        snu = disk.surface_density(rnu, t)
        ax1.plot(rnu, snu, "o", alpha=0.25, c="blue", ms=10)
        ax1.axvspan(rnu, 5.0, alpha=0.1, color="k")

def draw_arrow(t0, t1, text):
    x0 = disk.radius_of_influence(t0)
    x1 = disk.radius_of_influence(t1)
    y0 = 0.06
    ax1.annotate(
        "",
        xy=(x0, y0),
        xytext=(x1, y0),
        arrowprops=dict(arrowstyle="<-", color="k"),
        ha="left",
        va="center",
    )
    ax1.text(0.5 * (x0 + x1), y0, text, ha="center", va="bottom")

draw_arrow(-8, -4, r"$r_{\nu}(t)$")
ax1.set_xlim(0.5, 5.0)
ax1.set_xlabel(r"Radius $[r_{\rm dec}]$")
ax1.set_ylabel(r"Surface density $\Sigma$")
show()
```

## Contributing
Contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
