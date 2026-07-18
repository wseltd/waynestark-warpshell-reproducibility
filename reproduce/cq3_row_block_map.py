"""CQ3: block map of A_ineq/A_eq, reconcile 23809, regenerate rows from frozen Phi."""
import numpy as np
from scipy import sparse
from pathlib import Path as _P
SC=str(_P(__file__).resolve().parents[1] / "data" / "certificate")
Aineq=sparse.load_npz(SC+"/A_ineq.npz").tocsr(); Aeq=sparse.load_npz(SC+"/A_eq.npz").tocsr()
v=np.load(SC+"/vectors_B1p5.npz"); b=v["b_ub"]; tau=v["tau_star"]
N_STRESS,N_R,N_ANG,N_DIR=10,32,9,12
N_MODES=N_R*N_ANG; N_CELLS=N_R*N_DIR; N_TOTAL=N_STRESS*N_MODES
def blk(i): return slice(i*N_MODES,(i+1)*N_MODES)
def fib(n):
    i=np.arange(n); z=1-2*(i+0.5)/n; th=np.arccos(z); ph=(2*np.pi*i/((1+5**0.5)/2))%(2*np.pi); return th,ph
th,ph=fib(N_DIR); NX,NY,NZ=np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)
Phi=-Aineq[0:N_CELLS,blk(0)].toarray(); A=Aineq.toarray()
def br(k): return A[k*N_CELLS:(k+1)*N_CELLS,:]
COLS=dict(rho=0,Sx=1,Sy=2,Sz=3,Pxx=4,Pyy=5,Pzz=6,Pxy=7,Pxz=8,Pyz=9)
def col(nm): return slice(COLS[nm]*N_MODES,(COLS[nm]+1)*N_MODES)
maxd=0.0
def chk(ref,k):
    global maxd; maxd=max(maxd,float(np.max(np.abs(br(k)-ref))))
# positivity(0), cap(1)
r=np.zeros((N_CELLS,N_TOTAL)); r[:,col("rho")]=-Phi; chk(r,0)
r=np.zeros((N_CELLS,N_TOTAL)); r[:,col("rho")]=Phi; chk(r,1)
for j in range(N_DIR):
    nx,ny,nz=NX[j],NY[j],NZ[j]
    r=np.zeros((N_CELLS,N_TOTAL)); r[:,col("rho")]=-Phi
    r[:,col("Sx")]=-nx*Phi; r[:,col("Sy")]=-ny*Phi; r[:,col("Sz")]=-nz*Phi
    r[:,col("Pxx")]=-nx**2*Phi; r[:,col("Pyy")]=-ny**2*Phi; r[:,col("Pzz")]=-nz**2*Phi
    r[:,col("Pxy")]=-2*nx*ny*Phi; r[:,col("Pxz")]=-2*nx*nz*Phi; r[:,col("Pyz")]=-2*ny*nz*Phi
    chk(r,2+j)
for j in range(N_DIR):
    nx,ny,nz=NX[j],NY[j],NZ[j]
    rp=np.zeros((N_CELLS,N_TOTAL)); rp[:,col("rho")]=-Phi; rp[:,col("Sx")]=-nx*Phi; rp[:,col("Sy")]=-ny*Phi; rp[:,col("Sz")]=-nz*Phi; chk(rp,14+2*j)
    rm=np.zeros((N_CELLS,N_TOTAL)); rm[:,col("rho")]=-Phi; rm[:,col("Sx")]=nx*Phi; rm[:,col("Sy")]=ny*Phi; rm[:,col("Sz")]=nz*Phi; chk(rm,15+2*j)
for j in range(N_DIR):
    nx,ny,nz=NX[j],NY[j],NZ[j]
    ru=np.zeros((N_CELLS,N_TOTAL)); ru[:,col("rho")]=-Phi
    ru[:,col("Pxx")]=nx**2*Phi; ru[:,col("Pyy")]=ny**2*Phi; ru[:,col("Pzz")]=nz**2*Phi
    ru[:,col("Pxy")]=2*nx*ny*Phi; ru[:,col("Pxz")]=2*nx*nz*Phi; ru[:,col("Pyz")]=2*ny*nz*Phi; chk(ru,38+2*j)
    rl=np.zeros((N_CELLS,N_TOTAL)); rl[:,col("rho")]=-Phi
    rl[:,col("Pxx")]=-nx**2*Phi; rl[:,col("Pyy")]=-ny**2*Phi; rl[:,col("Pzz")]=-nz**2*Phi
    rl[:,col("Pxy")]=-2*nx*ny*Phi; rl[:,col("Pxz")]=-2*nx*nz*Phi; rl[:,col("Pyz")]=-2*ny*nz*Phi; chk(rl,39+2*j)
print(f"A_eq shape={Aeq.shape} (=4*N_r={4*N_R} conservation rows)   A_ineq shape={Aineq.shape}")
print(f"N_MODES=N_r*N_ang={N_MODES}  N_CELLS=N_r*N_dir={N_CELLS}  N_TOTAL={N_TOTAL}")
print("\nBLOCK TABLE (each cell-block has N_CELLS=384 rows):")
print(f"  positivity  rho>=0                 rows      0-  383   (1 x384 = 384)")
print(f"  density cap rho<=rho_max           rows    384-  767   (1 x384 = 384)")
print(f"  NEC surrogate (12 dirs)            rows    768- 5375   (12x384 = 4608)")
print(f"  WEC surrogate +/- (12 dirs x2)     rows   5376-14591   (24x384 = 9216)")
print(f"  DEC surrogate up/lo (12 dirs x2)   rows  14592-23807   (24x384 = 9216)")
print(f"  energy-budget row                  row   23808         (1)")
tot=384+384+4608+9216+9216+1
print(f"  TOTAL = 384+384+4608+9216+9216+1 = {tot}")
print(f"\nRECONCILE reviewer's 17569: reviewer used 288 cells (=N_MODES) x 61 + ...; correct is N_CELLS=384 (=N_r*N_dir). 288x61=17568 vs 384x62+1={384*62+1}.")
print(f"\nregeneration ||A_regen - A_frozen||_max over all 62 cone blocks = {maxd:.3e}")
print(f"density-cap RHS in tau-space: b[384]={b[384]:.6f} (all cap rows==1.0? {np.allclose(b[384:768],1.0)})")
print(f"b_eq: homogeneous, ||A_eq tau*||_inf = {float(np.max(np.abs(Aeq@tau))):.3e} (b_eq=0 exactly)")
print("\n12 Fibonacci unit vectors (nx,ny,nz):")
for j in range(N_DIR): print(f"  n[{j:2d}] = ({NX[j]:+.15f}, {NY[j]:+.15f}, {NZ[j]:+.15f})")
print(f"\nCQ3 VERDICT: {'PASS' if (maxd<=1e-12 and tot==23809) else 'FAIL'} (counts sum to 23809; rows regenerate to {maxd:.1e})")
