# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:41:16 2016

@author: ARKP
"""

import matplotlib.pyplot as plt
import numpy as np

plt.subplot(311, title='Happiness as a function of $a$')
w=10; c=1; a=np.arange(0,10,.1)
U = -np.exp(-(w - c*a**2))
plt.plot(a, U)

plt.subplot(312, title='Happiness as a function of $w$')
w=np.arange(0,10,.1); c=1; a=2
U = -np.exp(-(w - c*a**2))
plt.plot(w, U)

plt.subplot(313, title='Happiness as a function of $c$')
w=10; c=np.arange(0,10,.1); a=2
U = -np.exp(-(w - c*a**2))
plt.plot(c, U)

plt.tight_layout()

def optimize_work(t=2, c=0.5):
    '''For a given value of t and c, searches the space from s = 0% to s = 100%,
    plotting the level of work the employee will choose to maximize his happiness'''
    out = []
    for s in np.arange(0,1,.01):       # Search s-space
        a = np.arange(0,10,.01)        # Worker tries every level of work-ethic
        w = (t + s * a).clip(0)        # Worker sees how much he makes for that amount of work
                                       # Note: clip(0) ensures that pay is non-negative
        U = -np.exp(-(w - c * a**2))   # Worker weighs work against income to evaluate happiness
        idx = np.argmax(U)             # Worker picks work-ethic that maximizes happiness 
        out.append((s, a[idx], U[idx], w[idx]))
    s,a,U,w = zip(*out)
    plt.plot(s, a, label='t = {}'.format(t), alpha=.5)
    plt.legend(loc='best'); plt.xlabel('Commission rate'); plt.ylabel('Work')
    
optimize_work(t=2.0);
optimize_work(t=10.0);
plt.title('Work by commision rate for employee with $c=0.5$');

def optimize_profit(t=0, c=0.5, plot=True):
    '''For a given value of t and c, searches the space from s = 0% to s = 100%,
    plots the profit that the firm will generate based on predicted employee behavior'''
    out = []
    for s in np.arange(0,1,.01):       # Search s-space
        a = np.arange(0,10,.01)        # Worker tries every level of work-ethic
        w = (t + s * a).clip(0)        # Worker sees how much he makes for that amount of work
                                       # Note: clip(0) ensures that pay is non-negative
        U = -np.exp(-(w - c * a**2))   # Worker weighs work against income to evaluate happiness
        idx = np.argmax(U)             # Worker picks work-ethic that maximizes happiness 
        out.append((s, a[idx], U[idx], w[idx]))
    s,a,U,w = zip(*out)
    profit = np.array(a) - np.array(w) # Firm calculates profit across s-space acocunting for worker behavior
    if plot:
        plt.plot(s, profit, label='$t = {},\, c = {}$'.format(t, c))
        plt.legend(loc='best'); plt.xlabel('Commission rate'); plt.ylabel('Profit')
    optimized_rate = s[np.argmax(profit)]
    plt.axvline(optimized_rate, ls='--', lw=1, color='black', )
    plt.axhline(0, lw=1, color='black')
    return optimized_rate
    
optimize_profit(.2)
optimize_profit(.1)
optimize_profit(0.)
plt.title('Profit by commision rate for employee with $c=0.5$');

def optimize_happiness(t=0, c=0.5):
    '''For a given value of t and c, searches the space from s = 0% to s = 100%,
    plotting the maximum happiness possible at that value of s'''
    out = []
    for s in np.arange(0,1,.01):       # Search s-space
        a = np.arange(0,10,.01)        # Worker tries every level of work-ethic
        w = (t + s * a).clip(0)        # Worker sees how much he makes for that amount of work
                                       # Note: clip(0) ensures that pay is non-negative
        U = -np.exp(-(w - c * a**2))   # Worker weighs work against income to evaluate happiness
        idx = np.argmax(U)             # Worker picks work-ethic that maximizes happiness 
        out.append((s, a[idx], U[idx], w[idx]))
    s,a,U,w = zip(*out)
    plt.plot(s, U, label='$t = {},\,c={}$'.format(t,c))
    plt.legend(loc='best'); plt.xlabel('Commission rate'); plt.ylabel('Happiness')

optimize_happiness(c=.1)
optimize_happiness(c=.5)
optimize_happiness(c=1.)
plt.title('Happiness by commision rate for employee with no base pay');

print 'Best commission rate when c=0.1 is {}'.format(optimize_profit(c=.1, plot=True))
print 'Best commission rate when c=0.5 is {}'.format(optimize_profit(c=.5, plot=True))
print 'Best commission rate when c=1.0 is {}'.format(optimize_profit(c=1., plot=True))
plt.title('Profit by commision rate (assuming no base pay)');