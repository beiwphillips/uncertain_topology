from MorseComplex import *
import matplotlib.pyplot as plt
import scipy.io

xComponent = scipy.io.loadmat('data/xComponent')
yComponent = scipy.io.loadmat('data/yComponent')
mandatoryMaxima = scipy.io.loadmat('data/mandatoryMax')

print(xComponent)
print('~'*80)
print(yComponent)
print('~'*80)
print(mandatoryMaxima)

mc = MorseComplex2D(xComponent, yComponent, mandatoryMaxima)
data = mc.getMandatoryFlowField()

print(data)