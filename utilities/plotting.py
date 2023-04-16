#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:52:14 2023

@author: lucreziacerto
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_state_values(V):

    def get_Z(x, y):
        if (x,y) in V:
            return V[x,y]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(0, 8)
        y_range = np.arange(0, 8)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=0.0, vmax=10.0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('State values HideAndSeek')
    get_figure(True, ax)
    plt.show()
    



def plot_policy(value_f):

    def get_Z(x, y):
        if (x,y) in value_f:
            return value_f[(x,y)]
        else:
            return 0

    def get_figure(ax):
        x_range = np.arange(0, 7,dtype=int)
        y_range = np.arange(0, 7)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y) for x in x_range] for y in y_range])
        max = np.max(Z)
        #extent=[10.5, 21.5, 0.5, 10.5]
        surf = ax.imshow(Z, cmap='Blues', vmin=0, vmax=max)
        plt.xticks(x_range)
        plt.yticks(y_range)
        #plt.gca().invert_yaxis()
        plt.title('Seeker strike location')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, cax=cax)
            
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    get_figure(ax)
    plt.show()
    
