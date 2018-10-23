## Inversion 101/201, as part of Geodynamics 101
# by Lars Gebraad, 2018


# The most useful Python packages ever made!
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt

# Our linear 1D forward model for bread
def flourToBreadRelationship(flour):
    bread = (800.0 / 500.0) * flour
    return bread

# Our complex non-linear 1D forward model of our bread
def complexFlourToBreadRelationship(flour):
    bread = (0.000353327 * (flour ** 7)
             - 1.97863 * (flour ** 6)
             + 4398.71 * (flour ** 5)
             - (4.93162 * 10 ** 6) * (flour ** 4)
             + (2.91197 * 10 ** 9) * (flour ** 3)
             - (8.47364 * 10 ** 11) * (flour ** 2)
             + (1.04124 * 10 ** 14) * (flour) - 1.83574 * 10 ** 14) / 6658527342880.0
    return bread

# Our complex non-linear 2D forward model for bread
def flourWaterToBreadRelationship(flour, water):
    factor = (np.exp((-1e-2) * np.abs(300 * flour - 500 * water) / np.sqrt(300 ** 2 + 500 ** 2)))
    bread = (800.0 / 500.0) * flour * factor
    return bread

# Our directly inverted 1D relationship
def breadToFlourRelationship(bread):
    flour = (800.0 / 500.0) * bread
    return flour

# Misfit functional of the complex forward function
def complexMisfit(flour, observation):
    return np.abs(complexFlourToBreadRelationship(flour) - observation)


## 1D model visualization
flourRange = np.arange(0, 700, 0.01)
plt.plot(flourRange, flourToBreadRelationship(flourRange)) #        < enable one of these two
# plt.plot(flourRange, complexFlourToBreadRelationship(flourRange)) # <  enable one of these two
plt.plot([0, 700], [500, 500], '--')

# Making the graph a bit nicer
plt.legend(['forward relationship', 'Observed data'])
plt.xlabel('flour [grams]')
plt.ylabel('bread [grams]')
plt.xlim([0, 700])
plt.grid()
plt.show()

## Misfit for complex non-linear 1D model
# plt.plot(flourRange, complexMisfit(flourRange, 500)) # < Enable this if you want to investigate the complex 1d relationship
# plt.xlabel('flour [grams]')
# plt.ylabel('misfit [grams]')
# plt.xlim([0, 700])
# plt.grid()
# plt.show()

## 2D model complete model space visualization (expensive!)
flour = np.arange(0, 1000.0, 1)
water = np.arange(0, 600.0, 1)
X, Y = np.meshgrid(flour, water)  # grid of point
Z = flourWaterToBreadRelationship(X, Y)  # evaluation of the function on the grid
im = plt.imshow(Z, cmap=plt.cm.gist_heat)  # drawing the function
plt.xlabel('flour [grams]')
plt.ylabel('water [grams]')
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label('bread [grams]')
plt.show()


## 2D Grid search
max = np.max(Z) # Remembering the original extent of data for the colorbar
min = np.min(Z) # Remembering the original extent of data for the colorbar
x, y = np.where(np.abs(Z - 500) > 50)
Z[x, y] = np.nan
im = plt.imshow(Z, cmap=plt.cm.gist_heat, vmin=min, vmax=max)  # drawing the function
plt.xlabel('flour [grams]')
plt.ylabel('water [grams]')
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label('bread [grams]')
plt.show()

## Probabilistic bread and inversions
plt.plot(flourRange, complexMisfit(flourRange, 500))
mu = 400
sigma = 12
plt.plot(flourRange, -np.log(mlab.normpdf(flourRange, mu, sigma)))
plt.plot(flourRange, -np.log(mlab.normpdf(flourRange, mu, sigma)) + complexMisfit(flourRange, 500))
plt.legend(['Original misfit', 'Prior knowledge misfit', 'Combined misfit'])
plt.xlabel('flour [grams]')
plt.ylabel('misfit [grams]')
plt.xlim([0, 700])
plt.ylim([0, 707])
plt.grid()
plt.show()

## 2D Bayesian analysis
flour = np.arange(0, 1000.0, 1)
water = np.arange(0, 600.0, 1)
X, Y = np.meshgrid(flour, water)  # grid of point
Z = np.abs(flourWaterToBreadRelationship(X, Y) - 500)  # evaluation of the function on the grid
W = (np.abs(((X - 500)**2  + (Y - 300)**2)**0.5 - 150)/0.25)
im = plt.imshow(Z, cmap=plt.cm.gist_heat_r)  # drawing the function
plt.xlabel('flour [grams]')
plt.ylabel('water [grams]')
plt.gca().invert_yaxis()
cb = plt.colorbar()
cb.set_label('misfit [grams]')
plt.show()