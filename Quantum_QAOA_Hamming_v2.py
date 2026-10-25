# Quantum_QAOA_Hamming_v2.py

# QAOA - Quantum Approximate Optimization Algorithm  
# QUBO - Quadratic Unconstrained Binary Optimization  

# matematičke transformacije:  
# Isingov Hamiltonijan, Hammingov uslov težine i logaritamska uzajamna informacija 


import pandas as pd
import numpy as np
import random
from scipy.optimize import minimize

# Čist Qiskit 1.x standard
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

MAX_KUBITA = 5  
BROJ_STANJA = 32
TOTAL_BROJEVI = 39

random.seed(39)
np.random.seed(39)


# Putanja do podataka (CSV)
putanja = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv'
# putanja = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv'
# putanja = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv'


df = pd.read_csv(putanja, header=None)

# Izolacija tačno poslednjih 7 kolona sa repa tabele (hronološki niz, sortiran)
H = df.iloc[:, -7:].values.astype(int)
for i in range(len(H)):
    H[i] = np.sort(H[i])
N_redova = len(H)

# 1. PREVAĐENJE U PROSTOR OFSETA (32 STANJA)
ofseti = np.zeros(N_redova, dtype=int)
for idx, red in enumerate(H):
    ofset = red[-1] - red[0]  # Najveći minus najmanji broj (čist skalar)
    ofseti[idx] = np.clip(ofset, 0, BROJ_STANJA - 1)

# Brojimo istorijska poklapanja za svako od 32 stanja unapred
istorijska_poklapanja_ofseta = np.zeros(BROJ_STANJA)
for o in ofseti:
    istorijska_poklapanja_ofseta[o] += 1

pojedinačna_verovatnoća = np.zeros(BROJ_STANJA)
zajednička_verovatnoća = np.zeros((BROJ_STANJA, BROJ_STANJA))

for o in ofseti:
    pojedinačna_verovatnoća[o] += 1.0

for t in range(N_redova - 1):
    i = ofseti[t]
    j = ofseti[t+1]
    zajednička_verovatnoća[i, j] += 1.0

pojedinačna_verovatnoća /= N_redova
zajednička_verovatnoća /= (N_redova - 1)

J_istorija = np.zeros((BROJ_STANJA, BROJ_STANJA))
for i in range(BROJ_STANJA):
    for j in range(BROJ_STANJA):
        if i != j and zajednička_verovatnoća[i, j] > 0:
            J_istorija[i, j] = -np.log(zajednička_verovatnoća[i, j] / (pojedinačna_verovatnoća[i] * pojedinačna_verovatnoća[j] + 1e-9))

# 2. FORMULISANJE MATRICE Q PREKO BINARNIH KORELACIJA
Q = np.zeros((MAX_KUBITA, MAX_KUBITA))
for s_i in range(BROJ_STANJA):
    bin_i = [(s_i >> k) & 1 for k in range(MAX_KUBITA)]
    
    E_auto = -np.log(pojedinačna_verovatnoća[s_i] + 1e-9) if pojedinačna_verovatnoća[s_i] > 0 else 0.0
    for k_i in range(MAX_KUBITA):
        if bin_i[k_i] == 1:
            Q[k_i, k_i] += E_auto

    for s_j in range(BROJ_STANJA):
        bin_j = [(s_j >> k) & 1 for k in range(MAX_KUBITA)]
        for k_i in range(MAX_KUBITA):
            if bin_i[k_i] == 1:
                for k_j in range(MAX_KUBITA):
                    if bin_j[k_j] == 1 and k_i != k_j:
                        Q[k_i, k_j] += J_istorija[s_i, s_j]

# Funkcija cilja sa strogim kaznenim penalom za ofsete bez poklapanja
def racunaj_energetsko_ocekivanje(parametri):
    gamma, beta = parametri
    qc = QuantumCircuit(MAX_KUBITA, MAX_KUBITA)
    
    for i in range(MAX_KUBITA):
        qc.h(i)
        
    for i in range(MAX_KUBITA):
        qc.rz(Q[i, i] * gamma, i)
    for i in range(MAX_KUBITA):
        for j in range(i + 1, MAX_KUBITA):
            qc.cx(i, j)
            qc.rz((Q[i, j] + Q[j, i]) * gamma, j)
            qc.cx(i, j)
            
    for i in range(MAX_KUBITA):
        qc.rx(2 * beta, i)
        
    qc.measure(range(MAX_KUBITA), range(MAX_KUBITA))
    
    simulator = AerSimulator(method='statevector')
    counts = simulator.run(qc, shots=1024).result().get_counts()
    
    ukupna_energija = 0.0
    for bitstring, count in counts.items():
        stanje_idx = int(bitstring, 2)
        
        # Ako je izmereni ofset nemoguć ili ima 0 poklapanja u bazi, udari ogroman kazneni penal!
        if istorijska_poklapanja_ofseta[stanje_idx] == 0:
            ukupna_energija += 5000.0 * count
        else:
            stanja = [int(b) for b in reversed(bitstring)]
            e = 0.0
            for i in range(MAX_KUBITA):
                if stanja[i] == 1:
                    e += Q[i, i]
                    for j in range(MAX_KUBITA):
                        if i != j and stanja[j] == 1:
                            e += Q[i, j]
            ukupna_energija += e * count
        
    return ukupna_energija / 1024

# 4. KLASIČNA VARIJACIONA OPTIMIZACIJA UGLOVA
pocetni_uglovi = [0.1, 0.1]
opt_rezultat = minimize(racunaj_energetsko_ocekivanje, pocetni_uglovi, method='COBYLA', options={'maxiter': 35})
optimalni_gamma, optimalni_beta = opt_rezultat.x

# 5. FINALNI KVANTNI SHOT
finalno_kolo = QuantumCircuit(MAX_KUBITA, MAX_KUBITA)
for i in range(MAX_KUBITA): finalno_kolo.h(i)
for i in range(MAX_KUBITA): finalno_kolo.rz(Q[i, i] * optimalni_gamma, i)
for i in range(MAX_KUBITA):
    for j in range(i + 1, MAX_KUBITA):
        finalno_kolo.cx(i, j)
        finalno_kolo.rz((Q[i, j] + Q[j, i]) * optimalni_gamma, j)
        finalno_kolo.cx(i, j)
for i in range(MAX_KUBITA): finalno_kolo.rx(2 * optimalni_beta, i)
finalno_kolo.measure(range(MAX_KUBITA), range(MAX_KUBITA))

simulator = AerSimulator(method='statevector')
final_counts = simulator.run(finalno_kolo, shots=2048).result().get_counts()

# Sortiram izmerena stanja po verovatnoći, ali filtriram ona koja nemaju poklapanja u istoriji
sortirani_bitstringovi = sorted(final_counts.items(), key=lambda item: item[1], reverse=True)

optimalni_ofset = 0
for bitstring, count in sortirani_bitstringovi:
    provera_ofseta = int(bitstring, 2)
    if istorijska_poklapanja_ofseta[provera_ofseta] > 0:
        optimalni_ofset = provera_ofseta
        break

# 6. REKONSTRUKCIJA NA OSNOVU REALNIH ISTORIJSKIH RASPODELA ZA TAJ OFSET
indeksi_poklapanja = [idx for idx, o in enumerate(ofseti) if o == optimalni_ofset]

prosecne_pozicije = np.mean(H[indeksi_poklapanja], axis=0)
kombinacija = [int(np.round(p)) for p in prosecne_pozicije]

kombinacija = [np.clip(b, 1, 39) for b in kombinacija]
def ocisti_duplikate(niz):
    unikatni = []
    for x in niz:
        while x in unikatni or x > 39:
            x = (x % 39) + 1
        unikatni.append(x)
    return sorted(unikatni)

finalna_kombinacija = ocisti_duplikate(kombinacija)

print()
print(f"Mod: Pravilna Ofset Redukcija na 5 Kubita (Strict Constraint Filter)")
print(f"Analizirano redova iz CSV: {N_redova}")
print(f"Kvantno izmeren optimalni prostorni ofset: {optimalni_ofset}")
print(f"Broj istorijskih poklapanja za ovaj ofset: {len(indeksi_poklapanja)}")
print(f"Predložena kombinacija (Ground State): {finalna_kombinacija}")
print()
"""
Mod: Pravilna Ofset Redukcija na 5 Kubita (Strict Constraint Filter)
Analizirano redova iz CSV: 4708
Kvantno izmeren optimalni prostorni ofset: 27
Broj istorijskih poklapanja za ovaj ofset: 247
Predložena kombinacija (Ground State): [7, 11, 15, 20, 25, 29, 34]



Mod: Pravilna Ofset Redukcija na 5 Kubita (Strict Constraint Filter)
Analizirano redova iz CSV: 2976
Kvantno izmeren optimalni prostorni ofset: 31
Broj istorijskih poklapanja za ovaj ofset: 1513
Predložena kombinacija (Ground State): [3, 9, 14, 20, 26, 31, 37]



Mod: Pravilna Ofset Redukcija na 5 Kubita (Strict Constraint Filter)
Analizirano redova iz CSV: 1732
Kvantno izmeren optimalni prostorni ofset: 13
Broj istorijskih poklapanja za ovaj ofset: 5
Predložena kombinacija (Ground State): [15, 18, 19, 21, 24, 27, 28]
"""
