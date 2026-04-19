# ================== TALL BUILDING APP (FULL VERSION) ==================

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from math import pi, sqrt

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")
AUTHOR_NAME = "Benyamin"

# ---------------- DATA ----------------
@dataclass
class Building:
    n_story: int
    H_story: float
    plan_x: float
    plan_y: float
    bay_x: float
    bay_y: float
    n_bays_x: int
    n_bays_y: int
    DL: float
    LL: float
    Ct: float
    x: float

# ---------------- CORE CALC ----------------
def run_analysis(b: Building):
    H = b.n_story * b.H_story
    A = b.plan_x * b.plan_y

    W = (b.DL + b.LL) * A * b.n_story
    M = W * 1000 / 9.81

    T_code = b.Ct * (H ** b.x)
    T_target = 0.95 * T_code
    K = 4 * pi**2 * M / T_target**2

    # MDOF
    n = b.n_story
    m = M / n
    Mmat = np.diag([m]*n)

    k_story = [K/n*(1.2 - 0.6*(i/n)) for i in range(n)]

    Kmat = np.zeros((n,n))
    for i in range(n):
        if i==0:
            Kmat[i,i]+=k_story[i]
        else:
            Kmat[i,i]+=k_story[i]
            Kmat[i,i-1]-=k_story[i]
            Kmat[i-1,i]-=k_story[i]
            Kmat[i-1,i-1]+=k_story[i]

    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Mmat)@Kmat)

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    periods = [2*pi/np.sqrt(w) for w in eigvals[:5]]

    return H, W, T_target, K, periods, eigvecs

# ---------------- MODE SHAPES ----------------
def plot_modes(H, n, eigvecs, periods):
    y = np.linspace(0, H, n)
    fig, ax = plt.subplots(1,5, figsize=(18,5))

    for i in range(5):
        phi = eigvecs[:,i]
        phi = phi/phi[-1]
        ax[i].plot(phi, y, marker='o')
        ax[i].set_title(f"Mode {i+1}\nT={periods[i]:.2f}s")
        ax[i].invert_yaxis()
        ax[i].grid()

    return fig

# ---------------- PLAN ----------------
def plot_plan(b: Building):
    fig, ax = plt.subplots(figsize=(6,6))

    # boundary
    ax.plot([0,b.plan_x,b.plan_x,0,0],
            [0,0,b.plan_y,b.plan_y,0])

    # grid
    for i in range(b.n_bays_x+1):
        x=i*b.bay_x
        ax.plot([x,x],[0,b.plan_y],alpha=0.2)

    for j in range(b.n_bays_y+1):
        y=j*b.bay_y
        ax.plot([0,b.plan_x],[y,y],alpha=0.2)

    # columns
    for i in range(b.n_bays_x+1):
        for j in range(b.n_bays_y+1):
            ax.add_patch(plt.Rectangle(
                (i*b.bay_x-0.3,j*b.bay_y-0.3),
                0.6,0.6))

    # core
    cx = b.plan_x/2
    cy = b.plan_y/2
    ax.add_patch(plt.Rectangle((cx-5,cy-5),10,10,
                 fill=False,linewidth=2))

    ax.set_aspect('equal')
    ax.set_title("Plan View")
    ax.grid()

    return fig

# ---------------- UI ----------------
st.title("Tall Building Structural Analysis")
st.write(f"Prepared by **{AUTHOR_NAME}**")

col1,col2 = st.columns(2)

with col1:
    n_story = st.number_input("Stories",10,100,50)
    H_story = st.number_input("Story Height",2.5,5.0,3.4)
    plan_x = st.number_input("Plan X",20.0,200.0,80.0)
    plan_y = st.number_input("Plan Y",20.0,200.0,80.0)

with col2:
    DL = st.number_input("Dead Load",1.0,15.0,6.5)
    LL = st.number_input("Live Load",0.5,10.0,2.5)
    Ct = st.number_input("Ct",0.01,0.1,0.0488)
    x = st.number_input("Exponent",0.5,1.0,0.75)

b = Building(n_story,H_story,plan_x,plan_y,10,10,8,8,DL,LL,Ct,x)

if st.button("Run Analysis"):

    H,W,T,K,periods,eigvecs = run_analysis(b)

    st.subheader("Results")
    st.write(f"Height = {H:.2f} m")
    st.write(f"Weight = {W:,.0f} kN")
    st.write(f"Target Period = {T:.2f} s")

    st.subheader("Modes")
    st.pyplot(plot_modes(H,n_story,eigvecs,periods))

    st.subheader("Plan")
    st.pyplot(plot_plan(b))

    # download
    df = pd.DataFrame({
        "Mode":[1,2,3,4,5],
        "Period":periods
    })

    st.download_button("Download Results CSV",
                       df.to_csv(index=False),
                       "results.csv")
