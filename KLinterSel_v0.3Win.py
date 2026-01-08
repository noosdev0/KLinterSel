'''
Author: ACR
Program Name: KLinterSel
Last Update Dec 2025: v0.3

'''

import os
import psutil
import logging
from multiprocessing import Pool
import re
import sys
import argparse
import copy
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
import csv
import matplotlib
# Si estamos en un .exe de PyInstaller, forzamos un backend interactivo
#if getattr(sys, "frozen", False):
#v0.3 Windows option for pyinstaller .exe to be able of showing figures under windows
if getattr(sys, "frozen", False) and sys.platform.startswith("win"):
    matplotlib.use("TkAgg")  # requiere Tkinter disponible

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#from scipy.special import rel_entr # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
#from scipy.stats import chi2
# v0.3
from scipy.stats import hypergeom, kstest, cramervonmises, chisquare


# =========================
# Constants
# =========================

CLOSE  = "close_to_uniform"
MEDIUM = "medium_deviation"
STRONG = "strong_deviation"
SCENARIOS = ["uniform", "center", "left", "right", "extremes", "empirical"]

# (opcional) orden canónico de métricas
METRIC_ORDER = ["ks_stat", "cvm_stat", "cv_counts"]

#Limitest pruebas falsos positivos
FP_RS_WARN  = 100_000  # empieza a ser largo
FP_RS_HARD  = 1_000_000 # probablemente inviable en la práctica


# Configurar el logging para escribir mensajes en un archivo
#logging.basicConfig(filename='memory.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
# ACR Nov 2025
logger = logging.getLogger()                 # logger raíz
logger.setLevel(logging.INFO)                # activar mensajes INFO

handler = logging.FileHandler(
    'KLinterSel.log',
    mode='a',
    delay=True                               # evita crear el archivo hasta el primer logging
)

formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == handler.baseFilename
           for h in logger.handlers):
    logger.addHandler(handler)

'''
KL distance distribution test
Python version >=  3.7.11 required.

Este programa lee 2 categorias de ficheros:

1) El fichero original de los datos genómicos que es un csv (comma or semicolon separated) o txt con la siguiente estructura:

CHR POS
1    12345
1    12367
1    230001
2    34
3    2405
3    19121785
.
.
.

La primera columna identifica el cromosoma y la segunda la posición del snp dentro del cromosoma. Los cromosomas deben
estar ordenados de menor a mayor igual que las posiciones.

2) Los ficheros de sitios significativos resultantes de cada método. Con la misma estructura que el fichero anterior
pero el programa puede para estos ficheros identificar algunos formatos adicionales según la extensión:

tped
hapflk
norm

Estos ficheros tienen que ser como mínimo 2, de modo que al menos podamos comparar dos métodos.
Si el usuario lo desea puede meter el mismo fichero dos veces como control (deberán salir todos los cromosomas significativos
y todas las posiciones como intersecciones).


La funcion genrandomdata(datos,datosig) genera un fichero de sitios significativos al azar a partir de la estructura datos
obtenida tras myutils.FilterCsv del fichero original de datos y con la estructura de los datos  datossig obtenida tras myutils.FilterCsv
del fichero de sitios significativos


USAGE:

if file names are totsnps.csv for the original data and the results files are sigsmethod1.csv sigsmethod2.csv sigsmethod3.csv

python3 KLinterSel.py totsnps.csv sigsmethod1.csv sigsmethod2.csv sigsmethod3.csv

is equivalent to 

python3 KLinterSel.py totsnps.csv sigsmethod1.csv sigsmethod2.csv sigsmethod3.csv --path . --SL 0.05 --dist 10000 --perm 10000

if the files are in /home/b/results/data

python3 KLinterSel.py totsnps.csv sigsmethod1.csv sigsmethod2.csv sigsmethod3.csv --path ./home/b/results/data

CONTROL WITH RANDOM CANDIDATES:

Add argument --rand
python3 KLinterSel.py totsnps.csv sigsmethod1.csv sigsmethod2.csv sigsmethod3.csv --rand

By including --rand, the files sigsmethod1.csv, sigsmethod2.csv, and sigsmethod3.csv are used only to count their number of candidates
and randomly sample them from totsnps.csv.

Add argument --notest
python3 KLinterSel.py totsnps.csv sigsmethod1.csv sigsmethod2.csv sigsmethod3.csv --notest --dist 10000

By including --notest, the files sigsmethod1.csv, sigsmethod2.csv, and sigsmethod3.csv are used only to calculate the intersection
between methods for all chromosomes for the given distance (1E4 by default).

v0.3

New Hypergeometric and uniform tests

'''
PARALLEL=True
MAXPROC=1
BLOCK_SIZE=100
ALFA=0.05
RANDOM=False
TEST=True
REMUESTREO=True
random_files = False # if true generates randomly the selective sites based on the numbers of the real significative data

#UNIFORME=False
KILOBASES=False
FIGURE=False
STATS=False
# thresholds for strict control of files with very similar candidates if the ratio min(n_i,n_j)/max(n_i,n_j)>=UMBRAL1 and
# UMBRAL2 % of snps in file with min(n_i,n_j) snps are included in file with max(n_i,n_j) then exclude the file with the lowest number of snps
UMBRAL1=0.75
UMBRAL2=0.95

numrs=0
factor_seguridad=0.8
MinPerm=50

def issorted(array):
    return np.all(array[:-1] <= array[1:])

def obtener_memoria_disponible_giB():
    memoria_disponible_bytes = psutil.virtual_memory().available
    memoria_disponible_gib = memoria_disponible_bytes / (1024 ** 3)
    return memoria_disponible_gib

def calcular_memoria(array_shape, dtype=np.float64):
    num_elementos = np.prod(array_shape)
    bytes_por_elemento = np.dtype(dtype).itemsize
    memoria_bytes = num_elementos * bytes_por_elemento
    memoria_gib = memoria_bytes / (1024 ** 3)  # Convertir a GiB
    return memoria_gib

def calcular_tamano_bloque(array_shape, dtype=np.float64, factor_seguridad=factor_seguridad):

    memoria_disponible_gib = obtener_memoria_disponible_giB()
    memoria_usada_por_array_gib = calcular_memoria(array_shape, dtype)

    # Ajustar con un factor de seguridad para evitar el uso completo de la memoria
    memoria_disponible_segura_gib = memoria_disponible_gib * factor_seguridad

    if memoria_requerida_gib >= memoria_disponible_segura_gib:

        # Calcular cuántos bloques de la forma (x, array_shape[1]) caben en la memoria disponible
        # Aquí x es lo que queremos calcular
        bytes_por_fila = array_shape[1] * np.dtype(dtype).itemsize #tamaño en bytes de una sola fila, array_shape[1]  es el número de columnas
        filas_por_bloque = int((memoria_disponible_segura_gib * 1024 ** 3) / bytes_por_fila)

        # Asegurarse de que el tamaño del bloque es al menos 1 y no excede el número total de filas
        filas_por_bloque = max(1, min(filas_por_bloque, array_shape[0]))
    else:
        filas_por_bloque=array_shape[0]

    return filas_por_bloque

def procesar_bloque(array_shape, dtype=np.float64, factor_seguridad=factor_seguridad):
    memoria_disponible_gib = obtener_memoria_disponible_giB()
    memoria_requerida_gib = calcular_memoria(array_shape, dtype)

    if memoria_requerida_gib >= factor_seguridad * memoria_disponible_gib:
        mensaje_error = f"Required memory: {memoria_requerida_gib:.2f} GiB, Available memory: {memoria_disponible_gib:.2f} GiB.\n"
        
        #raise MemoryError(mensaje_error)
        return False, mensaje_error, memoria_requerida_gib
    return True, '', memoria_requerida_gib

def procesar_replica(args):
    r, numfiles, datos_filtrados, Lsitios, c, maxsites = args
    rmuestra = []
    for i in range(1, numfiles):
        rmuestra.append(np.array(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)))
    return totdist(*rmuestra, end=maxsites)

def calcular_avertot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites,maxproc=MAXPROC, block_size=BLOCK_SIZE):
    '''
    Usage: avertot_distances = calcular_avertot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites)
    '''
    avertot_distances = np.zeros(maxsites)
    num_blocks = numrs // block_size + (1 if numrs % block_size else 0)

    for block_idx in range(num_blocks):
        print(f'Processing block {block_idx+1} from a total of {num_blocks}')
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, numrs)
        args_list = [(r, numfiles, datos_filtrados, Lsitios, c, maxsites)
                     for r in range(start_idx, end_idx)]

        with Pool(processes=maxproc) as pool:
            results = pool.map(procesar_replica, args_list)

        for result in results:
            avertot_distances += result

    avertot_distances /= numrs
    return avertot_distances

def procesar_replica_p(args):
    r, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian = args
    rmuestra = [
        np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)
        for i in range(1, numfiles)
    ]
    rtot_distances = totdist(*rmuestra, end=maxsites)
    condicion = RelEntr(rtot_distances, avertot_distances) >= arrKL[c] and np.percentile(rtot_distances, 50) <= emedian[c]
    return 1 if condicion else 0

##def calcular_arrP_paralelo(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=MAXPROC):
##    '''
##    Usage: arrP[c] = calcular_arrP_paralelo(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian)
##    '''
##    # Crear una lista de argumentos para cada réplica
##    args_list = [
##        (r, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian)
##        for r in range(numrs)
##    ]
##
##    # Usar multiprocessing para paralelizar
##    with Pool(processes=maxproc) as pool:
##        resultados = pool.map(procesar_replica_p, args_list)
##
##    # Calcular arrP[c]
##    #arrP = sum(resultados) / numrs
##    arrP = (sum(resultados)+1.0) / (numrs+1.0) #v0.3 corrección Davison–Hinkley
##    return arrP

def calcular_arrP_paralelo_bloques(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian, maxproc=MAXPROC, block_size=BLOCK_SIZE):
    '''
    Usage:
    arrP[c] =
    calcular_arrP_paralelo_bloques(
    numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=num_processes,block_size=BLOCK_SIZE)
    '''
    # Inicializar contador de réplicas que cumplen la condición
    count_satisfy_condition = 0

    # Calcular número de bloques
    num_blocks = numrs // block_size + (1 if numrs % block_size else 0)

    for block_idx in range(num_blocks):
        print(f'Processing block {block_idx+1} from a total of {num_blocks}')

        # Definir índices de inicio y fin para el bloque actual
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, numrs)

        # Crear lista de argumentos para el bloque actual
        args_list = [
            (r, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian)
            for r in range(start_idx, end_idx)
        ]

        # Procesar el bloque en paralelo
        with Pool(processes=maxproc) as pool:
            resultados = pool.map(procesar_replica_p, args_list)

        # Acumular resultados del bloque
        count_satisfy_condition += sum(resultados)

    # Calcular la proporción de réplicas que cumplen la condición
    #arrP = count_satisfy_condition / numrs
    arrP = (count_satisfy_condition +1.0) / (numrs + 1.0) #v0.3 corrección Davison–Hinkley
    return arrP

def genrandomdata(datos,datossig):
    # Generar la nueva estructura con valores muestreados aleatoriamente
    ndarrays_aleatorios = []

    for i, fila in enumerate(datossig):
        longitud = fila.size
        valores_aleatorios = np.random.choice(datos[i], size=longitud, replace=False)
        ndarrays_aleatorios.append(valores_aleatorios)

    # Convertir la lista de ndarrays a una lista de ndarrays
    ndarrays_aleatorios = [np.array(fila) for fila in ndarrays_aleatorios]

    return ndarrays_aleatorios
# Función para eliminar las comas de los números en un fichero de texto (no csv). Devuelve el nuevo nombre de fichero sin comas en los números
# Si no había comas devuelve el nombre del fichero original
def eliminar_comas(ruta,separator=' '):
    # Define el objeto de la ruta para poder separar la ruta y el nombre del fichero
    path = Path(ruta)
    try:
        commas=False

        # Leer el archivo CSV de entrada
        with open(ruta, mode='r', newline='') as file_entrada:

            file_entrada.readline()  # salta la cabecera

            # Check if there is commas in the values
            for row in file_entrada: # row es una lista de cadenas
                row=row.strip() # removes leading and trailing whitespaces including new line char
                if not row:  # Ignorar líneas en blanco
                    continue
                for elemento in row.split(sep=separator):
                    if elemento.find(',')>=0:
                        commas=True
                        break
                if commas:
                    break

            # Si se detectaron comas
            if commas: # genera el fichero sin comas en los valores

                lname=path.name.split(sep='.')
    
                archivo_salida=''.join(lname[0:-1])+'_nc.'+lname[-1]

                ruta_salida= path.parent / archivo_salida # El operador / de Path une de manera portable entre OS
        
                with open(ruta_salida, mode='w', newline='') as file_salida:


                    # Procesar cada fila del archivo de entrada
                    file_entrada.seek(0) # Volver al inicio del archivo

                    #Escribir cabecera
                    file_salida.write(file_entrada.readline() )

                    #Procesar el resto del fichero 

                    for row in file_entrada:
                        row=row.strip()
                        if not row:  # Ignorar líneas en blanco sin comas ni nada
                            continue
                        elif all(elemento == '' for elemento in row.split(sep=separator)):  # Ignorar líneas en blanco que contienene solo separador si este no es el blanco
                            continue
                        elif any(elemento.strip() == '' for elemento in row.split(sep=separator)): #Hay valores pero una celda en blanco
                            raise ValueError(f"In {eliminar_comas.__name__} a blank item was found in the row: {str(row)} when reading {path.name}")

                        # Eliminar las comas de los números en la fila
                        lrow=row.split(sep=separator)
                        fila_procesada = [valor.replace(',', '') for valor in lrow]
                        
                        # Escribir la fila procesada en el archivo de salida
                        file_salida.write(' '.join(fila_procesada)+'\n')
                        
                return archivo_salida
            return path.name
    except ValueError as ve:
        print(ve)
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nThe file {path.name} was not found.\n")
        sys.exit(1)
# Función para eliminar las comas de los números en un csv. Devuelve el nuevo nombre de fichero sin comas en los números
# Si no había comas devuelve el nombre del fichero original
def eliminar_comas_csv(ruta,delimiter, debug=False):

    # Define el objeto de la ruta para poder separar la ruta y el nombre del fichero
    path = Path(ruta)
    try:
        commas=False

        # Leer el archivo CSV de entrada
        with open(ruta, mode='r', newline='') as file_entrada:

            csv_reader = csv.reader(file_entrada, delimiter=delimiter)

            next(csv_reader, None)  # salta la cabecera

            # Check if there is commas in the values
            for row in csv_reader: # row es una lista de cadenas
                if not row:  # Ignorar líneas en blanco
                    continue
                   
                for elemento in row:
                    if elemento.find(',')>=0:
                        commas=True
                        break
                if commas:
                    break

            # Si se detectaron comas
            if commas: # genera el fichero sin comas en los valores

                archivo_salida=path.name.split(sep='.')[0]+'_nc.csv'

                ruta_salida= path.parent / archivo_salida # El operador / de Path une de manera portable entre OS
        
                with open(ruta_salida, mode='w', newline='') as file_salida:

                    csv_writer = csv.writer(file_salida, delimiter=delimiter)

                    # Procesar cada fila del archivo de entrada
                    file_entrada.seek(0) # Volver al inicio del archivo
                     # Recrear el objeto csv_reader
                    csv_reader = csv.reader(file_entrada, delimiter=delimiter)
                    cabecera = next(csv_reader)
                    csv_writer.writerow(cabecera)

                    #Procesar el resto del fichero 

                    for row in csv_reader:
                        if not row:  # Ignorar líneas en blanco sin comas ni nada
                            continue
                        elif all(elemento == '' for elemento in row):  # Ignorar líneas en blanco que contienene comas
                            continue
                        elif any(elemento.strip() == '' for elemento in row):
                            raise ValueError(f"In {eliminar_comas_csv.__name__} a blank item was found in the row: {str(row)} when reading {path.name}")

                        # Eliminar las comas de los números en la fila
                        fila_procesada = [str(int(valor.replace(',', ''))) for valor in row]
                        
                        # Escribir la fila procesada en el archivo de salida
                        csv_writer.writerow(fila_procesada)
                        
                return archivo_salida
            return path.name
    except ValueError as ve:
        # Si no es un número, devolver el valor original
        #print(f'Excepción de ValueError leyendo {path.name} en {eliminar_comas_csv.__name__}')
        print(ve)
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nThe file {path.name} was not found.\n")
        sys.exit(1)
    
def filter_clusters(array, D):
    if len(array) == 0:
        return array

    # Ordenar el array por si acaso
    #array = np.sort(array)

    clusters = []
    current_cluster = [array[0]]

    for i in range(1, len(array)):
        if array[i] - array[i-1] <= D:
            current_cluster.append(array[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [array[i]]

    # Añadir el último cluster
    if current_cluster:
        clusters.append(current_cluster)

    # Filtrar cada cluster: mantener solo el primero y el último valor
    filtered_values = []
    for cluster in clusters:
        if len(cluster) > 0:
            filtered_values.append(cluster[0])
            if len(cluster) > 1:
                filtered_values.append(cluster[-1])

    # Eliminar duplicados si hay algún cluster con un solo elemento
    filtered_values = sorted(list(set(filtered_values)))

    return np.array(filtered_values)

def FilterCsv(ruta, Kmax=np.inf, D=-np.inf):
    """ Abre un fichero csv con dos columnas la primera para el número de cromosoma y la segunda la posición
        y almacena en un array de numpy cada columna
        
        Returns: Una lista de arrays cada elemento (array) de la lista corresponde a un cromosoma
    """
    path = Path(ruta)
    try:

        # Diccionario para agrupar los valores de la segunda columna según los valores de la primera columna
        datos_agrupados = defaultdict(list)
       
        with open(ruta, mode='r', newline='') as csvfile:

            #identify wheter the csv file is comma or semicolon separated
            
            delimiter = csv.Sniffer().sniff(csvfile.readline()).delimiter
            # Tras leer el delimitador vuelve a poner el cursor a 0
            csvfile.seek(0)

            # Crear un objeto lector CSV
            #csv_reader = csv.DictReader(csvfile, delimiter=delimiter)
            csv_reader = csv.reader(csvfile, delimiter=delimiter)

            # Leer y descartar la cabecera
            next(csv_reader)

            # Leer y agrupar las filas
            
            for row in csv_reader:
                if all(elemento == '' for elemento in row):  # Ignorar líneas en blanco
                    continue
                elif any(elemento.strip() == '' for elemento in row):
                    raise ValueError(f"In {FilterCsv.__name__} a blank item was found in the row: {str(row)} when reading {path.name}")

                clave = int(row[0])  # Valor de la primera columna
                valor = int(row[1])  # Valor de la segunda columna
                datos_agrupados[clave].append(valor)

            # Determinar el número máximo de claves
            max_clave = max(datos_agrupados.keys())

        # Crear una lista de listas con longitudes diferentes, asegurando que todas las claves estén presentes
        lista_filtrada = [datos_agrupados[clave] if clave in datos_agrupados else [] for clave in range(1, max_clave + 1)]

        #Calcula el número total de SNPs antes del posible filtrado
        totsnps= np.sum([len(fila) for fila in lista_filtrada])

        #Convertir cada sublista a un ndarray
        ndarrays_filtrados = [np.array(fila) for fila in lista_filtrada]

        # 11 Julio 2025
        # Nueva funcionalidad: filtrar clusters si el tamaño del array es mayor que Kmax
        if not np.isinf(Kmax) and not np.isinf(D):
            filtered_ndarrays = []
            for array in ndarrays_filtrados: # para cada cromosoma
                if len(array) >= Kmax:
                    filtered_array = filter_clusters(array, D)
                    filtered_ndarrays.append(filtered_array)
                else:
                    filtered_ndarrays.append(array)
                
            return filtered_ndarrays, totsnps
        else:
            return ndarrays_filtrados, totsnps
    
    except ValueError as valerr:
        print(valerr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nThe file {path.name} was not found.\n")
        sys.exit(1)

def filter_crompos(ruta, cromcol=0, poscol=1, Kmax=np.inf, D=-np.inf, header=True):
    """ Abre un fichero y almacena en un array de numpy la columna con el identificador de cromosoma y la de la posición
        
        Returns: Una lista de arrays cada elemento (array) de la lista corresponde a un cromosoma y dentro de cada array están las posiciones
        Si el número de posiciones es mayor de Kmax entonces identifica clusters de posiciones que estén todos dentro de la distancia D
        y se queda solo con los extremos
        
    """
    path=Path(ruta)
    try:

        # Diccionario para agrupar los valores de la columna poscol según los valores de la columna cromcol
        datos_agrupados = defaultdict(list)
       
        with open(ruta, mode='r', newline='') as file:
            
            if header:
                # Leer y descartar la cabecera
                file.readline()

            # Leer y agrupar las filas
            
            for row in file:
                row=row.strip()
                if not row:  # Ignorar líneas en blanco
                    continue
                listafila=row.split()
                clave = int(listafila[cromcol])  # Valor de la columna que identifica cromosoma (irá desde 1 hasta el máximo número de cromosomas que haya)
                valor = int(listafila[poscol])  # Valor de la columna que identifica posición del snp
                datos_agrupados[clave].append(valor) # diccionario donde cada cromosoma es una clave que lleva una lista de posiciones como valor

            # Determinar el número máximo de claves
            max_clave = max(datos_agrupados.keys()) if datos_agrupados else 0

        # Crear una lista de listas con longitudes diferentes, asegurando que todas las claves estén presentes incluso aunque no estén en el fichero
        #es decir si el cromosoma 4 no tiene SNPs y no aparece se añade una lista vacía []
        lista_filtrada = [datos_agrupados[clave] if clave in datos_agrupados else [] for clave in range(1, max_clave + 1)]

        #Calcula el número total de SNPs antes del posible filtrado
        totsnps= np.sum([len(fila) for fila in lista_filtrada])

        #Convertir cada sublista a un ndarray
        ndarrays_filtrados = [np.array(fila) for fila in lista_filtrada]

        #Check positions are sorted
        for array in ndarrays_filtrados:
            if array.size == 0:
                next
            else:
                if not issorted(array):
                    array.sort()
                    
        # Nueva funcionalidad: filtrar clusters si el tamaño del array es mayor que Kmax
        
        if not np.isinf(Kmax) and not np.isinf(D):

            filtered_ndarrays = []
            
            for array in ndarrays_filtrados: # para cada cromosoma
                if len(array) >= Kmax:
                    filtered_array = filter_clusters(array, D)
                    filtered_ndarrays.append(filtered_array)
                else:
                    filtered_ndarrays.append(array)
            return filtered_ndarrays,totsnps
        else:
            return ndarrays_filtrados, totsnps
            
        #return ndarrays_filtrados
        #return filtered_ndarrays
    
    except ValueError as valerr:
        print(valerr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nThe file {path.name} was not found.\n")
        sys.exit(1)

def filter_norm(ruta,cromid=1, poscol=1, critcol=-1, criter=1, Kmax=np.inf, D=-np.inf):
    """ Abre un fichero y almacena en un array de numpy las posiciones cuya columna critcol cumplen el criterio criter (>=criter)
        El formato del fichero .norm debe ser
        id	pos	gpos	p1	ihh1	p2	ihh2	xpehh	normxpehh	crit
        Se asume un único cromosoma y por defecto se filtran las posiciones que tengan valor >=criter en la última columna (critcol=-1)
        Returns:
        Una lista de 1 array que corresponde al único cromosoma y los elementos del array son  las posiciones que tenían valor crit>=criter
    """
    path=Path(ruta)
    try:

        # Diccionario para agrupar los valores de la columna poscol según los valores de la columna cromcol:
        # en el caso de norm solo hay un cromosoma pero se deja así para que pueda extenderse a más cromosomas
        datos_agrupados = defaultdict(list) # Un diccionario vacío {} pero cuyas claves inexistentes se inicializan automáticamente a list() en vez de dar error
       
        with open(ruta, mode='r', newline='') as file:

            # Leer y descartar la cabecera
            file.readline()

            # Leer y agrupar las filas
            
            for row in file:
                row=row.strip()
                if not row:  # Ignorar líneas en blanco
                    continue
                listafila=row.split()
                clave = cromid  # Valor de la columna que identifica cromosoma (irá desde 1 hasta el máximo número de cromosomas que haya)
                # Condición para considerar la fila como posición significativa
                if int(listafila[critcol])>=criter:
                    valor = int(listafila[poscol])  # Valor de la columna que identifica posición del snp
                    datos_agrupados[clave].append(valor) # diccionario donde cada cromosoma es una clave que lleva una lista de posiciones como valor

            # Determinar el número máximo de claves, en el caso de norm siempre es 1
            max_clave = 1

        # Crear una lista de listas con longitudes diferentes, asegurando que todas las claves estén presentes incluso aunque no estén en el fichero
        #es decir si el cromosoma 4 no tiene SNPs y no aparece se añade una lista vacía []
        lista_filtrada = [datos_agrupados[clave] if clave in datos_agrupados else [] for clave in range(1, max_clave + 1)]

        #Calcula el número total de SNPs antes del posible filtrado
        totsnps= np.sum([len(fila) for fila in lista_filtrada])

        #Convertir cada sublista a un ndarray
        ndarrays_filtrados = [np.array(fila) for fila in lista_filtrada]

        #Check positions are sorted
        for array in ndarrays_filtrados:
            if array.size == 0:
                next
            else:
                if not issorted(array):
                    array.sort()

        # Nueva funcionalidad: filtrar clusters si el tamaño del array es mayor que Kmax
        if not np.isinf(Kmax) and not np.isinf(D):
            filtered_ndarrays = []
            for array in ndarrays_filtrados: # para cada cromosoma
                if len(array) >= Kmax:
                    filtered_array = filter_clusters(array, D)
                    filtered_ndarrays.append(filtered_array)
                else:
                    filtered_ndarrays.append(array)
                
            return filtered_ndarrays, totsnps
        else:
            return ndarrays_filtrados, totsnps
        
    
    except ValueError as valerr:
        print(valerr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nThe file {path.name} was not found.\n")
        sys.exit(1)

def mkdir(ruta):
    """ si no existe crea el directorio en la ruta indicada. Devuelve la ruta
        completa al directorio
    """
    ruta=ruta.strip() # limpia los espacios en blanco al inicio y fin

    path = Path(ruta)

    if(path.is_dir()): #ya existe
        return ruta
    elif(path.is_file()):
        print(f'A file named {path.name} already exists, the directory will not be created')
        return ""        
    # Crea todos los directorios parentales necesarios si no existen, si el directorio ya existe no lanza error
    path.mkdir(parents=True, exist_ok=True) #Requiere from pathlib import Path

    return ruta

# v0.2 Nov 2025. Versión más eficiente
def RelEntr(arr1, arr2):
    """
    KL-like discrepancy between two non-negative vectors arr1 and arr2.
    Normaliza ambos vectores para que sumen 1.
    Devuelve 0.0 en casos degenerados (una sola distancia, suma = 0, etc.).
    """
    a1 = np.asarray(arr1, dtype=float)
    a2 = np.asarray(arr2, dtype=float)

    if a1.size != a2.size:
        raise ValueError("Arrays must have the same length")

    # Con menos de 2 distancias el KL-like no aporta nada
    if a1.size < 2:
        return 0.0

    sum1 = a1.sum()
    sum2 = a2.sum()

    # Si alguno suma 0, no se puede normalizar => perfil degenerado
    if sum1 <= 0.0 or sum2 <= 0.0:
        return 0.0

    P = a1 / sum1
    Q = a2 / sum2

    # Solo aportan las posiciones donde P > 0
    mask = P > 0.0
    if not np.any(mask):
        return 0.0

    # Evitar problemas cuando Q es muy pequeño o 0
    eps = 1e-15
    Q_safe = np.clip(Q[mask], eps, None)

    ratio = P[mask] / Q_safe
    T = np.sum(P[mask] * np.log(ratio))

    return float(T)

def calculate_vectorized(rtot_distances, avertot_distances, arrKL, emedian, c):
    '''
    Calcula los valores p de manera vectorizada.
    Usage:  arrP_result = calculate_vectorized(rtot_distances, avertot_distances, arrKL, emedian, c)

    '''
    numrs = len(rtot_distances)
    # Convertir la lista de ndarrays a un solo ndarray 3D
    # Asumiendo que todos los subarrays tienen la misma longitud
    #rtot_array = np.array(rtot_distances)

    # Calcular percentiles y condiciones
    conditions_met = np.array([
        RelEntr(rtot_distances[r], avertot_distances) >= arrKL[c] and
        np.percentile(rtot_distances[r], 50) <= emedian[c]
        for r in range(numrs)
    ])

    # Contar cuántas condiciones se cumplen
    arrP_increment = np.sum(conditions_met)
    
    #arrP_increment / numrs

    return (arrP_increment+1) / (numrs+1) #v0.3 corrección Davison–Hinkley

def totdist(*arrays, ini=0, end=-1):
    """
    Calcula las distancias de los elementos del primer array con todos los siguientes,
    del segundo con todos los siguientes etc. Utilizando operaciones vectorizadas de NumPy.
    Si arrays = [np.array([1, 3, 7]), np.array([2, 4]), np.array([2, 3, 4, 7, 10])] obtenemos
    3x(2+5) + 2x5=31 distancias en general len(i)*[len(i+1)+len(i+2) +..+] + len(i+1)*[len(i+2)+len(i+3)+..+]
    Args:
        *arrays: colección de arrays de numpy

    Returns:
        np.ndarray: Un array de numpy que contiene todas las distancias ordenadas de menor a mayor.
    """
    # Inicializar una lista para almacenar todas las diferencias
    all_differences = []

    # Iterar sobre cada array
    for i in range(len(arrays)-1):

        # Obtener el array actual si no está vacío
        if arrays[i].size:
            current_array = arrays[i]


            # Iterar sobre todos los arrays siguientes
            for j in range(i + 1, len(arrays)):
                # Obtener el siguiente array
                next_array = arrays[j]

                # Calcular las diferencias absolutas entre cada elemento del array actual y cada elemento del siguiente array
                differences = np.abs(current_array[:, np.newaxis] - next_array)

                # Aplanar el array de diferencias y añadirlo a la lista de todas las diferencias
                # Usamos extend() para añadir cada una de las diferencias calculadas como elementos individuales a la lista all_differences
                all_differences.extend(differences.flatten())
        else:
            continue

    # Convertir la lista de todas las diferencias a un array de numpy y ordenarlo
    if end==-1 or end>len(all_differences):
        end = len(all_differences)
    return np.sort(np.array(all_differences))[ini:end]

def generate_rtot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites):
    '''
    Usage: rtot_distances = generate_rtot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites)
    '''
    # Usar una comprensión de listas para generar rtot_distances sin un bucle explícito
    rtot_distances = [
        totdist(*[
            np.array(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False))
            for i in range(1, numfiles)
        ], end=maxsites)
        for _ in range(numrs)
    ]

    return rtot_distances

def generate_uniform_rtot_distances(numrs, numfiles, Lsitios, c, maxsites, min_val, max_val):
    """
    Genera numrs réplicas de distancias totales asumiendo que las posiciones
    de los SNPs candidatos en cada fichero se distribuyen de forma uniforme
    entre min_val y max_val (inclusive).
    """
    rtot_distances = [
        totdist(*[
            np.random.randint(
                low=min_val,
                high=max_val + 1,
                size=Lsitios[i][c]
            )
            for i in range(1, numfiles)
        ], end=maxsites)
        for _ in range(numrs)
    ]
    return rtot_distances

def calculate_arrP(rtot_distances, avertot_distances, arrKL, emedian, c, kl_divs):
    '''
    Calcula el valor p dados el array numrs de distancias
    kl_divs se obtuve desde RelEntr_vectorized(rtot_distances, avertot_distances)
    arrP_result = calculate_arrP(rtot_distances, avertot_distances, arrKL, emedian, c, kl_divs)
    '''
    numrs = len(rtot_distances)

    # Calcular el percentil 50 para cada conjunto de datos en rtot_distances
    median_values = np.array([np.percentile(rtot_distances[r], 50) for r in range(numrs)])

    # Evaluar las condiciones para cada conjunto de datos
    condition1 = kl_divs >= arrKL[c]
    condition2 = median_values <= emedian[c]

    # Combinar condiciones
    conditions_met = condition1 & condition2

    # Contar cuántas condiciones se cumplen
    arrP_increment = np.sum(conditions_met)

    #return arrP_increment / numrs
    return (arrP_increment+1.0) / (numrs+1.0) #v0.3 corrección Davison–Hinkley



'''
numdists calcula para un cromosoma dado el número de distancias posibles que hay entre los distintos arrays de resultados.

'''
def numdists(*arrays):
    numarrays=len(arrays)
    D=0
    for i in range(numarrays-1):
        d=0
        for j in range(i+1,numarrays):
            d+=len(arrays[j])
        D+=len(arrays[i])*d
    return D


#Crea un diccionario con tantas claves como combinaciones posibles de n elementos tomados de 2 o más: Para n=3 C(3,2)+C(3,3) = 4
#Los nombres de las claves son por defecto 'A12','A13' etc pero se puede cambiar con el argumento keyname
def create_combination_dict(n, keyname='A'):
    elements = [f'{keyname}{i+1}' for i in range(n)]
    combination_dict = {}

    ini=len(keyname)
    
    for r in range(2, n + 1):
        for combo in combinations(elements, r):
            # Extraer los números de los elementos y concatenarlos
            numbers = [elem[ini:] for elem in combo]
            key = keyname + ''.join(numbers)
            combination_dict[key] = []

    return combination_dict

'''
Función que recorre las claves AiAj del diccionario cd desde la primera i y hasta la penultima i,
tal que para cada par AiAj[par] comprobar si AiAj[par][0] está encualquier par de AiAk[algunpar][0] (k mayor que j) y
si es cierto entonces hacer cd[AiAjAk].append(np.array(AiAj[par][0],AiAj[par][1] AiAk[algunpar][1]))

Notar que a partir de Python 3.7, los diccionarios mantienen el orden de inserción.
'''
def intersec_Dn(*arrays, D=np.inf, keyname='A_'):

    '''
    NOT USED CHANGED BY intersec_Dn_Opt_no_sort
    
    USAGE example n=5:
    arrays = [
        np.array([1, 2, 3, 4, 5]),
        np.array([3, 4, 5, 6]),
        np.array([5, 6, 7, 8, 9, 10]),
        np.array([6]),
        np.array([7])
    ]
    D=1

    cd= intersec_Dn(*arrays, D=D, keyname='SEL_')


    '''

    n = len(arrays)
    cd = {}

    # Generar todas las combinaciones posibles de índices de arrays
    print(f'Generate all possible combinations of positions between methods that are at a distance <= {D}.')
    for length in range(2, n + 1):
        print(f'{n} choose {length}')
        for indices in combinations(range(n), length):
            # Construir la clave del diccionario
            key = keyname + ''.join(map(str, [i + 1 for i in sorted(indices)]))

            # Inicializar la lista para esta clave
            cd[key] = []

            # Para pares, comparar directamente
            if length == 2:
                i, j = indices
                for elem_i in arrays[i]:
                    for elem_j in arrays[j]:
                        if abs(elem_i - elem_j) <= D:
                            cd[key].append(np.array([elem_i, elem_j]))

            else:
                # Para combinaciones de más de dos arrays
                # Recorrer todas las combinaciones de longitud length-1 dentro de esta combinación
                for sub_indices in combinations(indices, length - 1):
                    sub_key = keyname + ''.join(map(str, [i + 1 for i in sorted(sub_indices)]))

                    if sub_key in cd:
                        for sub_combination in cd[sub_key]:
                            # Buscar en el array restante de la combinación
                            remaining_index = [i for i in indices if i not in sub_indices][0]
                            for elem in arrays[remaining_index]:
                                # Verificar la condición de distancia con al menos un elemento de la subcombinación
                                if any(abs(sub_elem - elem) <= D for sub_elem in sub_combination):
                                    # Crear una nueva combinación asegurando el orden correcto
                                    new_combination = np.empty(length, dtype=sub_combination.dtype)
                                    sub_combination_iter = iter(sub_combination)
                                    for pos, idx in enumerate(sorted(indices)):
                                        if idx in sub_indices:
                                            new_combination[pos] = next(sub_combination_iter)
                                        else:
                                            new_combination[pos] = elem

                                    # Verificar que la combinación no esté ya en la lista
                                    if not any(np.array_equal(new_combination, x) for x in cd[key]):
                                        cd[key].append(new_combination)

    return cd

def intersec_Dn_Opt_no_sort(*arrays, D=np.inf, keyname='A_'):
    n_arrays = len(arrays)
    cd = {}
    kset=set()
    print(f'Generate all possible combinations of positions between methods that are at a distance <= {D}.')
    for length in range(2, n_arrays + 1):
        print(f'{n_arrays} choose {length}')
        for indices in combinations(range(n_arrays), length):
            indices_sorted = sorted(indices)
            key = keyname + ''.join(map(str, [i + 1 for i in indices_sorted]))
            unique_combinations = set()

            if length == 2:
                i, j = indices_sorted
                array_i, array_j = arrays[i], arrays[j]
                for elem_i in array_i:
                    for elem_j in array_j:
                        if abs(elem_i - elem_j) <= D:
                            combo_tuple = tuple([elem_i, elem_j])  # Sin ordenar
                            unique_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in unique_combinations]
                # ACR Nov 2025
                if length == n_arrays:  # esto solo puede ocurrir si n_arrays == 2
                    for combo in unique_combinations:
                        # combo es una tupla con (elem_i, elem_j)
                        kset.update(combo)                
            else:
                temp_combinations = set()
                for sub_indices in combinations(indices_sorted, length - 1):
                    sub_key = keyname + ''.join(map(str, [i + 1 for i in sub_indices]))
                    if sub_key in cd:
                        for sub_combination in cd[sub_key]:
                            remaining_index = [i for i in indices_sorted if i not in sub_indices][0]
                            array_rest = arrays[remaining_index]
                            sub_combination_array = np.array(sub_combination)
                            for elem in array_rest:
                                distances = np.abs(sub_combination_array - elem)
                                if np.any(distances <= D):
                                    new_combination = np.empty(length, dtype=sub_combination.dtype)
                                    sub_combination_iter = iter(sub_combination)
                                    for pos, idx in enumerate(indices_sorted):
                                        if idx in sub_indices:
                                            new_combination[pos] = next(sub_combination_iter)
                                        else:
                                            new_combination[pos] = elem
                                    #acr jul 2025
                                    if length == n_arrays:
                                        #kset.add(elem)
                                        kset.update(new_combination.tolist()) # v0.2 Nov 2025
                                    combo_tuple = tuple(new_combination)
                                    temp_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in temp_combinations]
        
            
    return cd, kset

# v0.2 ACR Nov 2025 Sorted version of the previous intersec_Dn_Opt_no_sort

def intersec_Dn_Opt_sorted(*arrays, D=np.inf, keyname='A_'):
    n_arrays = len(arrays)
    cd = {}
    kset=set()
    #print(f'Generate all possible combinations of positions between methods that are at a distance <= {D}.')
    for length in range(2, n_arrays + 1):
        #print(f'{n_arrays} choose {length}')
        for indices in combinations(range(n_arrays), length):
            indices_sorted = sorted(indices)
            key = keyname + ''.join(map(str, [i + 1 for i in indices_sorted]))
            unique_combinations = set()

            if length == 2:
                i, j = indices_sorted
                array_i, array_j = arrays[i], arrays[j]
                for elem_i in array_i:
                    for elem_j in array_j:
                        if abs(elem_i - elem_j) <= D:
                            combo_tuple = tuple([elem_i, elem_j])  # Sin ordenar
                            unique_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in sorted(unique_combinations)]
                # ACR Nov 2025
                if length == n_arrays:  # esto solo puede ocurrir si n_arrays == 2
                    for combo in unique_combinations:
                        # combo es una tupla con (elem_i, elem_j)
                        kset.update(combo)
            else:
                temp_combinations = set()
                for sub_indices in combinations(indices_sorted, length - 1):
                    sub_key = keyname + ''.join(map(str, [i + 1 for i in sub_indices]))
                    if sub_key in cd:
                        for sub_combination in cd[sub_key]:
                            remaining_index = [i for i in indices_sorted if i not in sub_indices][0]
                            array_rest = arrays[remaining_index]
                            sub_combination_array = np.array(sub_combination)
                            for elem in array_rest:
                                distances = np.abs(sub_combination_array - elem)
                                if np.any(distances <= D):
                                    new_combination = np.empty(length, dtype=sub_combination.dtype)
                                    sub_combination_iter = iter(sub_combination)
                                    for pos, idx in enumerate(indices_sorted):
                                        if idx in sub_indices:
                                            new_combination[pos] = next(sub_combination_iter)
                                        else:
                                            new_combination[pos] = elem
                                    #acr jul 2025
                                    if length == n_arrays:
                                        #kset.add(elem)
                                        kset.update(new_combination.tolist()) # v0.2 Nov 2025
                                    combo_tuple = tuple(new_combination)
                                    temp_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in sorted(temp_combinations)]
        
            
    return cd, kset

# Función para comparar dos diccionarios de resultados
def compare_diccionaries(cd1, cd2):
    if cd1.keys() != cd2.keys():
        print("Las claves de los diccionarios no coinciden.")
        print("Claves en cd1:", cd1.keys())
        print("Claves en cd2:", cd2.keys())
        return False

    for key in cd1.keys():
        list1 = cd1[key]
        list2 = cd2[key]

        # Convertir cada array en una tupla ordenada para comparación
        set1 = set(tuple(sorted(arr)) for arr in list1)
        set2 = set(tuple(sorted(arr)) for arr in list2)

        if set1 != set2:
            print(f"Las intersecciones para la clave {key} no coinciden.")
            print("En cd1:", sorted(list1))
            print("En cd2:", sorted(list2))
            return False

    return True
'''
FIGURE HANDLING
'''
# Devuelve el número de bins y su ancho
def calculate_bins(data, min_bins=30, max_bins=500):
    n = len(data)
    if n == 0:
        return 1

    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        data_range = 1
        return min_bins,1

    # Usar la regla de Freedman-Diaconis
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1
    fd_width = (2 * iqr) / np.power(n, 1/3)
    if fd_width == 0:
        return min_bins,1
    num_bins = int(np.ceil(data_range / fd_width))

    # Limitar el número máximo de bins (ajustado para datos grandes)
    
    num_bins = min(num_bins, max_bins)  # Asegurar como máximo max_bins
    num_bins = max(num_bins, min_bins)  # Asegurar al menos min_bins

    return num_bins,  np.ceil(data_range / num_bins)

def determine_unit(max_value, manual_unit=None):
    if manual_unit == 'kb':
        return 'kb', 1e3
    elif manual_unit == 'Mb':
        return 'Mb', 1e6
    else:  # selección automática
        if max_value < 1e6:  # Menos de 1 millón → usar kb
            return 'kb', 1e3
        else:  # 1 millón o más → usar Mb
            return 'Mb', 1e6

def get_nice_scale(max_value, unit_divisor=1e6,fixed_max_value=None, target_ticks=10):
    candidate_units = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]  # en las unidades seleccionadas (kb o Mb)
    if fixed_max_value is not None:
        max_value_in_units = fixed_max_value / unit_divisor
    else:
        max_value_in_units = max_value / unit_divisor

    best_unit = None
    best_diff = float('inf')

    for unit in candidate_units:
        num_ticks = int(max_value_in_units / unit) + 1
        diff = abs(num_ticks - target_ticks)
        if diff < best_diff:
            best_diff = diff
            best_unit = unit
        elif diff == best_diff:
            if unit > best_unit:  # Preferir unidades más grandes para menos ticks si hay empate
                best_unit = unit

    return best_unit * unit_divisor  # Devolver la unidad en la escala original

def get_ticks(max_value, unit, fixed_max_value=None):

    if fixed_max_value is not None:
        max_val = fixed_max_value
    else:
        max_val = max_value
    
    ticks = np.arange(0, max_value + unit, unit)
    ticks = ticks[ticks <= max_value]
    return ticks

def format_tick_labels(ticks, unit_divisor):
    labels = []
    for tick in ticks:
        tick_in_units = tick / unit_divisor
        if tick_in_units >= 1:
            label = f'{tick_in_units:.0f}'
        elif tick_in_units >= 0.1:
            label = f'{tick_in_units:.1f}'
        else:
            label = f'{tick_in_units:.2f}'
        labels.append(label)
    return labels
'''
END FIGURE HANDLING
'''

def positive_int(value):
    """
    Validar que el valor proporcionado sea un entero mayor o igual que uno.
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} must be greater than or equal to 1")
    return ivalue


''' v 0.2 Nov 2025

Funciones para el filtrado estricto si los snps candidatos de uno de los ficheros coincide totalmente con el de otro

Uso:
    datos_filtrados_filtrados, eliminados = filtrar_datos(datos_filtrados, umbral=0.75)

    print("Candidatos eliminados (índices):", eliminados)

'''

def total_snps_file(file_list):
    """
    file_list es la lista de cromosomas del fichero i.
    Cada elemento es un np.array de SNPs.
    """
    return sum(arr.size for arr in file_list)


def todos_snps_en(file_origen, file_destino):
    """
    Devuelve True si TODOS los SNPs de file_origen están incluidos 
    en el correspondiente cromosoma de file_destino.
    file_origen y file_destino son listas paralelas.
    """
    for snps_i, snps_j in zip(file_origen, file_destino):

        if snps_i.size == 0:
            continue  # nada que comprobar

        if snps_j.size == 0:
            return False  # origen tiene SNPs y destino no

        # Inclusión vectorizada con numpy
        if not np.isin(snps_i, snps_j).all():
            return False

    return True


def snps_incluidos_umbral(file_origen, file_destino, umbral2=1.0):
    """
    Devuelve True si al menos una fracción 'umbral2' de los SNPs de file_origen 
    están contenidos en file_destino (mismo índice de cromosoma).
    
    file_origen y file_destino son listas paralelas de np.array.
    umbral2 ∈ (0,1]
    """
    total_snps_origen = 0
    total_incluidos = 0

    for snps_i, snps_j in zip(file_origen, file_destino):

        # Número total de SNPs del origen en este cromosoma
        ni = snps_i.size
        if ni == 0:
            continue

        total_snps_origen += ni

        # Si el destino está vacío, no puede incluir ninguno
        if snps_j.size == 0:
            continue

        # Vectorizado: booleanos indicando qué SNPs de i están en j
        incluidos = np.isin(snps_i, snps_j).sum()

        total_incluidos += incluidos

    # Si no hay SNPs, consideramos inclusión completa
    if total_snps_origen == 0:
        return True

    # Cálculo de fracción incluida
    fraccion = total_incluidos / total_snps_origen

    return fraccion >= umbral2

def filtrar_datos(datos_filtrados, umbral=0.75):

    numfiles = len(datos_filtrados)

    # total de SNPs por fichero
    n_tot = np.array([total_snps_file(df) for df in datos_filtrados])

    eliminados = defaultdict(list)
    ya_eliminados = set()

    for i in range(1, numfiles):
        if i in ya_eliminados:
            continue

        for j in range(i + 1, numfiles):
            if j in ya_eliminados:
                continue

            ni = n_tot[i]
            nj = n_tot[j]

            if ni == 0 and nj == 0:
                continue

            if ni <= nj:
                if nj == 0:
                    continue
                ratio = ni / nj
                #if ratio >= umbral and todos_snps_en(datos_filtrados[i], datos_filtrados[j]):
                if ratio >= umbral and snps_incluidos_umbral(datos_filtrados[i], datos_filtrados[j], umbral2=umbral):
                    eliminados[j].append(i)
                    ya_eliminados.add(i)
                    break

            else:  # ni > nj
                if ni == 0:
                    continue
                ratio = nj / ni
                #if ratio >= umbral and todos_snps_en(datos_filtrados[j], datos_filtrados[i]):
                if ratio >= umbral and snps_incluidos_umbral(datos_filtrados[j], datos_filtrados[i], umbral2=umbral):
                    eliminados[i].append(j)
                    ya_eliminados.add(j)
                    # no rompemos: i puede eliminar más candidatos

    nueva_lista = [
        f for idx, f in enumerate(datos_filtrados) if idx not in ya_eliminados
    ]

    return nueva_lista, eliminados


'''
v0.2 Nov 2025
Aplicamos las reglas por cromosoma lo que es un poco  más exigente y preciso.
En el caso global sería posible que en un cromosoma coincidan 99% pero en otro solo el 50% y según sea el umbral por ejemplo 75% podrían considerarse que cumple
cuando algún cromosoma no cumple ni de lejos. 

'''

def cromosomas_ok_B(file_i, file_j, umbral1=0.75, umbral2=1.0, min_snps=5):
    """
    Regla combinada B (ratio + inclusión) por cromosoma, simétrica:

    Para cada cromosoma c (file_i[c], file_j[c]):

      n_ic = len(file_i[c])
      n_jc = len(file_j[c])

      Si max(n_ic, n_jc) < min_snps -> se ignora el cromosoma porque son muy pocos snps y los solapamientos totales no son tan raros.

      Si max(n_ic, n_jc) >= min_snps:
          small = array del fichero con menos SNPs en ese cromosoma
          large = array del fichero con más SNPs en ese cromosoma

          Regla 1 local:
              small.size / large.size >= umbral1

          Regla 2 local:
              frac = (# SNPs de small que están en large) / small.size
              frac >= umbral2

    Devuelve True si:
      - existe al menos un cromosoma usable, y
      - en todos los cromosomas usables se cumplen ambas reglas.
    """
    crom_usables = 0

    for snps_i, snps_j in zip(file_i, file_j):
        n_i = snps_i.size
        n_j = snps_j.size

        max_n = max(n_i, n_j)
        if max_n < min_snps:
            continue  # cromosoma ignorado

        crom_usables += 1

        # si uno de los dos está vacío pero el otro no, y max_n>=min_snps, ya falla
        if n_i == 0 or n_j == 0:
            return False

        if n_i <= n_j:
            small = snps_i
            large = snps_j
            n_small = n_i
            n_large = n_j
        else:
            small = snps_j
            large = snps_i
            n_small = n_j
            n_large = n_i

        # Regla 1 local: ratio de tamaños
        if (n_small / n_large) < umbral1:
            return False

        # Regla 2 local: inclusión de small en large
        incluidos = np.isin(small, large).sum()
        frac = incluidos / n_small
        if frac < umbral2:
            return False

    # Si no hubo ningún cromosoma usable, no nos fiamos
    if crom_usables == 0:
        return False

    return True

def filtrar_datos_B(datos_filtrados, umbral1=0.75, umbral2=1.0, min_snps=5):
    """
    Filtrado usando la regla combinada B por cromosoma.

    Para cada par (i, j) con 1 <= i < j < numfiles:

      Si cromosomas_ok_B(datos_filtrados[i], datos_filtrados[j], ...) es True:
          => consideramos que uno de los ficheros es redundante.

        Aproximación: eliminamos el que tenga menos SNPs totales.
        (si empatan, se elimina el de índice mayor para mantener el de índice menor).

    Devuelve:
        nueva_lista, eliminados

    donde:
      nueva_lista es la lista de ficheros filtrados
      eliminados es un dict: superset_idx => [subset_idx eliminados]
    """
    numfiles = len(datos_filtrados)
    n_tot = np.array([total_snps_file(df) for df in datos_filtrados])

    eliminados = defaultdict(list)
    ya_eliminados = set()

    for i in range(1, numfiles):
        if i in ya_eliminados:
            continue

        for j in range(i + 1, numfiles):
            if j in ya_eliminados:
                continue

            # Regla combinada B por cromosoma
            if not cromosomas_ok_B(datos_filtrados[i], datos_filtrados[j],
                                   umbral1=umbral1, umbral2=umbral2,
                                   min_snps=min_snps):
                continue

            ni = n_tot[i]
            nj = n_tot[j]

            # Decidimos cuál eliminar
            if ni < nj:
                idx_small, idx_large = i, j
            elif nj < ni:
                idx_small, idx_large = j, i
            else:
                # Si empatan en total, por ejemplo nos quedamos con el de índice menor
                idx_small, idx_large = j, i  # elimina j

            eliminados[idx_large].append(idx_small)
            ya_eliminados.add(idx_small)

            if idx_small == i:
                break  # i ya se elimina, pasamos al siguiente i

    nueva_lista = [
        f for idx, f in enumerate(datos_filtrados) if idx not in ya_eliminados
    ]

    return nueva_lista, eliminados

# v0.2 Nov 2025
def format_arrP(x, dec_fijo=6, dec_sci=6):
    x = float(x)

    if x == 0.0:
        # Cero exacto → formato fijo
        return f"{x:.{dec_fijo}f}"

    if abs(x) >= 1e-6:
        # Notación normal
        return f"{x:.{dec_fijo}f}"
    else:
        # Notación científica
        return f"{x:.{dec_sci}e}"

def format_pvalue(p, nperm=10000):
    # número de decimales necesarios
    # ejemplo: 10000 → 4 decimales, 100000 → 5 decimales
    if p <= 0.0: #v0.3 Davison–Hinkley corrected p-value
        p=1.0/(nperm+1)
    dec = len(str(nperm)) - 1
    return f"{p:.{dec}f}"

def fmt_p(p, precision=4):
    return np.format_float_positional(float(p), precision=precision)



'''
***********************************************
v0.3 December 2025
Add a new parametric test:
Hypergeometric k-way intersection test (HGkI)
***********************************************
USAGE:
res = hypergeom_kway_windows_test(
    datos_filtrados=datos_filtrados,
    D=W,
    chrom_index=None,
    method_indices=list(range(1, numfiles)),  # k métodos
)

for row in res["per_chrom"]:
    print(row["chrom"], row["N_windows"], row["n_i"], row["k_obs"], row["p_value"])


'''
def _windows_count_from_range(pos_min, pos_max, D) -> int:
    """Legacy helper (not used in the current hypergeometric universe definition)."""
    """
    This function could be used if the data consist in genome-length universe (assembly based).
    Number of disjoint windows of size D covering [pos_min, pos_max] inclusive.
    """
    if pos_max < pos_min:
        return 0
    return int(np.ceil((pos_max - pos_min + 1) / D))


def _positions_to_windows(pos, pos_min, D) -> np.ndarray:
    """
    Map genomic positions to 0-based window indices using the effective range origin pos_min.
    window_id = floor((pos - pos_min) / D)
    """
    if pos.size == 0: # no snps in the chromosome or no candidates
        return pos.astype(np.int64) # return an empty array of integers
    # Ensure integer arithmetic
    pos = np.asarray(pos, dtype=np.int64)
    return (pos - pos_min) // D


def _kway_intersection_size(window_sets) -> int:
    """
    Exact k-way intersection size among sets in window_sets.
    """
    if not window_sets:
        return 0
    inter = window_sets[0].copy()
    for s in window_sets[1:]:
        inter.intersection_update(s)
        if not inter:
            return 0
    return len(inter)


def hypergeom_kway_intersection_pvalue(
    n_list, #: list[int],
    N,#: int,
    k_obs,#: int,
    *,
    reorder_by_size = True,
) -> float:
    """
    Exact cascade hypergeometric p-value for k-way intersection size >= k_obs.

    Parameters
    ----------
    n_list : list[int]
        Sizes n_i = |A_i| for each method i (in windows).
    N : int
        Universe size (number of windows).
    k_obs : int
        Observed k-way intersection size.
    reorder_by_size : bool
        If True, reorder methods by increasing n_i for efficiency (does not change result).

    Returns
    -------
    p_value : float
        Right-tail probability P(K_k >= k_obs) under the hypergeometric cascade null.
    """
    n_list = [int(n) for n in n_list]

    if N <= 0:
        # No universe: only possible if everything is empty
        return 1.0 if k_obs <= 0 else 0.0

    if any(n < 0 or n > N for n in n_list):
        raise ValueError("All n_i must satisfy 0 <= n_i <= N.")

    k = len(n_list)
    if k < 2:
        # With 0 or 1 method, the notion of k-way overlap test is trivial.
        return 1.0

    # Observed intersection cannot exceed min(n_i)
    n_min = min(n_list)
    if k_obs > n_min:
        return 0.0
    if k_obs <= 0:
        return 1.0

    if reorder_by_size:
        n_list = sorted(n_list)

    # DP distribution for K_1 is degenerate at n1
    n1 = n_list[0]
    prev_support = np.array([n1], dtype=np.int64)
    prev_pmf = np.array([1.0], dtype=np.float64)

    # Iterate j = 2..k
    for j in range(1, k):
        nj = n_list[j]

        # New support is 0..min(nj, max(prev_support))
        t_max = int(prev_support.max())
        new_max = min(nj, t_max)
        new_support = np.arange(new_max + 1, dtype=np.int64)
        new_pmf = np.zeros(new_support.size, dtype=np.float64)

        # For each possible previous intersection size t with probability prev_pmf,
        # convolve with Hypergeom(N, t, nj)
        for t, p_t in zip(prev_support, prev_pmf):
            if p_t == 0.0:
                continue
            t = int(t)
            if t == 0:
                # Then K_j must be 0
                new_pmf[0] += p_t
                continue

            x_max = min(t, nj)
            # Only x in [0, x_max] are feasible
            x = np.arange(x_max + 1, dtype=np.int64)
            # SciPy hypergeom: M=N, n=t successes, N=nj draws
            pmf_x = hypergeom.pmf(x, N, t, nj)
            new_pmf[: x_max + 1] += p_t * pmf_x

        # Numerical cleanup: renormalize defensively
        s = new_pmf.sum()
        if s <= 0.0:
            return 0.0
        new_pmf /= s

        # Compress support to non-zeros to speed next step
        nz = new_pmf > 0.0
        prev_support = new_support[nz]
        prev_pmf = new_pmf[nz]

    # Right-tail p-value from final distribution
    # prev_support holds the support values of K_k with their probabilities prev_pmf
    mask = prev_support >= k_obs
    return float(prev_pmf[mask].sum())



def hypergeom_kway_intersection_pvalue_stable(
    n_list, #: list[int],
    N,#: int,
    k_obs,#: int,
    *,
    reorder_by_size = True,
) -> float:
    """
    Exact cascade hypergeometric p-value for k-way intersection size >= k_obs.

    Parameters
    ----------
    n_list : list[int]
        Sizes n_i = |A_i| for each method i (in windows).
    N : int
        Universe size (number of windows).
    k_obs : int
        Observed k-way intersection size.
    reorder_by_size : bool
        If True, reorder methods by increasing n_i for efficiency (does not change result).

    Returns
    -------
    p_value : float
        Right-tail probability P(K_k >= k_obs) under the hypergeometric cascade null.
    """
    n_list = [int(n) for n in n_list]

    if N <= 0:
        # No universe: only possible if everything is empty
        return 1.0 if k_obs <= 0 else 0.0

    if any(n < 0 or n > N for n in n_list):
        raise ValueError("All n_i must satisfy 0 <= n_i <= N.")

    k = len(n_list)
    if k < 2:
        # With 0 or 1 method, the notion of k-way overlap test is trivial.
        return 1.0

    # Observed intersection cannot exceed min(n_i)
    n_min = min(n_list)
    if k_obs > n_min:
        return 0.0
    if k_obs <= 0:
        return 1.0

    if reorder_by_size:
        n_list = sorted(n_list)

    # DP distribution for K_1 is degenerate at n1
    n1 = n_list[0]
    prev_support = np.array([n1], dtype=np.int64)
    prev_pmf = np.array([1.0], dtype=np.float64)

    # Iterate j = 2..k
    for j in range(1, k):
        nj = n_list[j]

        # New support is 0..min(nj, max(prev_support))
        t_max = int(prev_support.max())
        new_max = min(nj, t_max)
        new_support = np.arange(new_max + 1, dtype=np.int64)
        new_pmf = np.zeros(new_support.size, dtype=np.float64)

        # For each possible previous intersection size t with probability prev_pmf,
        # convolve with Hypergeom(N, t, nj)
        for t, p_t in zip(prev_support, prev_pmf):
            if p_t == 0.0:
                continue
            t = int(t)
            if t == 0:
                # Then K_j must be 0
                new_pmf[0] += p_t
                continue

            x_max = min(t, nj)
            # Only x in [0, x_max] are feasible
            x = np.arange(x_max + 1, dtype=np.int64)
            # cálculo estable de la pmf hipergeométrica
            logpmf_x = hypergeom.logpmf(x, N, t, nj)
            # Shift para evitar underflow (no cambia la distribución tras renormalizar)
            m = np.max(logpmf_x)
            pmf_x = np.exp(logpmf_x - m)

            # IMPORTANT: renormalize so that sum_x pmf_x == 1 for this (N,t,nj)
            s_pmf = pmf_x.sum()
            if s_pmf > 0.0:
                pmf_x /= s_pmf
            else:
                # extremely unlikely fallback
                pmf_x[:] = 0.0
                pmf_x[0] = 1.0

            # acumulación ponderada
            new_pmf[: x_max + 1] += p_t * pmf_x

        # Numerical cleanup: renormalize defensively
        s = new_pmf.sum()
        if s <= 0.0:
            return 0.0
        new_pmf /= s

        # Compress support to non-zeros to speed next step
        thr = new_pmf.max() * np.finfo(np.float64).tiny
        nz = new_pmf > thr
        prev_support = new_support[nz]
        prev_pmf = new_pmf[nz]

    # Right-tail p-value from final distribution
    # prev_support holds the support values of K_k with their probabilities prev_pmf
    mask = prev_support >= k_obs
    return float(prev_pmf[mask].sum())



def hypergeom_kway_windows_test(
    datos_filtrados,#: list[list[np.ndarray]],
    D,#: int,
    *,
    chrom_index = None, # : int | None = None
    method_indices = None, #: list[int] | None = None,
    use_effective_range_from_data0 = True,
    reorder_by_size = True,
) -> dict:
    """
    Compute the Hypergeometric k-way intersection test per chromosome using disjoint windows of size D.

    This is designed to plug into KLinterSel's data structure:
      datos_filtrados[file_index][chrom] = np.array(positions)

    Convention in KLinterSel_v0.2.py:
      - datos_filtrados[0] is the ORIGINAL SNP dataset (all SNP positions).
      - datos_filtrados[1..] are candidate sets for each method.  :contentReference[oaicite:1]{index=1}

    Parameters
    ----------
    datos_filtrados : list[list[np.ndarray]]
        List over files; each file is a list over chromosomes of np arrays of positions.
    D : int
        Window size in bp (typically 1).
    chrom_index : int | None
        If provided, test only that 0-based chromosome index. Otherwise test all chromosomes.
    method_indices : list[int] | None
        Indices of method files to include (default: all files except 0).
        Example: [1,2,3,4] for 4 methods.
    use_effective_range_from_data0 : bool
        If True, define windows over [min(datos_filtrados[0][c]), max(datos_filtrados[0][c])].
        If False, you must provide your own logic for pos_min/pos_max (not implemented here).
    reorder_by_size : bool
        If True, reorder n_i increasing inside the cascade for efficiency.

    Returns
    -------
    results : dict
        {
          "per_chrom": [
              {
                "chrom": c+1,
                "N_windows": N,
                "n_i": [n1,...,nk],
                "k_obs": k_obs,
                "p_value": p,
                "warnings": warnings
                "missing": False
              }, ...
          ]
        }
    """
    if D <= 0:
        raise ValueError("D must be a positive integer (window size).")

    numfiles = len(datos_filtrados)
    if numfiles < 3:
        raise ValueError("Need at least original data file + >=2 method files.")

    numcroms = len(datos_filtrados[0])

    if method_indices is None:
        method_indices = list(range(1, numfiles))
    if len(method_indices) < 2:
        raise ValueError("Need at least 2 methods for a k-way intersection test.")

    # Basic validation
    for idx in method_indices:
        if idx <= 0 or idx >= numfiles:
            raise ValueError(f"Invalid method index {idx}. Must be in [1, {numfiles-1}].")

    chroms = [chrom_index] if chrom_index is not None else list(range(numcroms))

    per_chrom = []
    for c in chroms:
        warnings = []
        missing_flag = False
        # Define effective range from original data (recommended)
        if use_effective_range_from_data0:
            data0 = np.asarray(datos_filtrados[0][c], dtype=np.int64)
            if data0.size == 0:
                # No SNPs in this chromosome in the original dataset
                per_chrom.append({
                    "chrom": c + 1,
                    "N_windows": 0,
                    "n_i": [0 for _ in method_indices],
                    "k_obs": 0,
                    "p_value": 1.0,
                    "warnings": [],
                    "missing": False
                })
                continue
            pos_min = int(data0.min())
            pos_max = int(data0.max())
        else:
            raise NotImplementedError("Provide custom pos_min/pos_max logic if not using data0 effective range.")

        # --- Define the universe as selectable units under the null ---
        if D == 1:
            # SNP-level universe (exact overlap): universe = SNPs in original dataset
            U = set(np.unique(data0).tolist())   # positions as discrete units
            N = len(U)

            window_sets = []
            n_list = []
            for idx in method_indices:
                pos = np.asarray(datos_filtrados[idx][c], dtype=np.int64)

                cand = set(pos.tolist())
                missing = cand - U
                if missing:
                    missing_flag = True
                    warnings.append(
                        f"Method {idx}: {len(missing)} candidate SNP(s) not present in the "
                        "original SNP universe and ignored "
                        f"(e.g. {sorted(missing)[0]})."
                    )
                    
                s = cand & U
                window_sets.append(s)
                n_list.append(len(s))


        else:
            # Window-level universe (regional overlap): universe = occupied windows only
            all_win = _positions_to_windows(data0, pos_min, D)
            U = set(np.unique(all_win).tolist())  # occupied window IDs
            N = len(U)

            window_sets = []
            n_list = []
            for idx in method_indices:
                pos = np.asarray(datos_filtrados[idx][c], dtype=np.int64)
                win = _positions_to_windows(pos, pos_min, D)
                missing_win = set(win.tolist()) - U
                if missing_win:
                    missing_flag = True
                    example = sorted(missing_win)[0]
                    warnings.append(
                        f"Method {idx}: {len(missing_win)} candidate window(s) not present in the "
                        "occupied-window universe and ignored."
                        f"(e.g. window {example})."
                        f"This indicates candidate SNPs not present in the original SNP set."
                    )
                s = set(win.tolist()) & U
                window_sets.append(s)
                n_list.append(len(s))
        # ------------------------------------------------------------------------------

        

        if N <= 1:
            warnings.append(
                "Degenerate hypergeometric universe (N_windows <= 1). "
                "The p-value is necessarily 1."
            )
        elif N < 10:
            warnings.append(
                "Very small hypergeometric universe (N_windows < 10). "
                "The test may have very low resolution."
            )
        elif N < 20:
            warnings.append(
                "Small hypergeometric universe (N_windows < 20). "
                "Interpret results with caution."
            )


        k_obs = _kway_intersection_size(window_sets)

        p = hypergeom_kway_intersection_pvalue_stable(
            n_list=n_list,
            N=N,
            k_obs=k_obs,
            reorder_by_size=reorder_by_size,
        )

        per_chrom.append({
            "chrom": c + 1,
            "N_windows": int(N),
            "n_i": [int(x) for x in n_list],
            "k_obs": int(k_obs),
            "p_value": float(p),
            "warnings": warnings,
            "missing": missing_flag
        })

    return {"per_chrom": per_chrom}


'''
End of v0.3 with HGkI test added
'''

'''
***********************************************
v0.3 December 2025
Add a new ranking test for deviation from uniform:
Spatial Uniformity rank metrics (SURM)
***********************************************
'''

def _choose_bins(n, target_per_bin=8, min_bins=20, max_bins=200):
    """Choose number of bins based on SNP count n."""
    if n <= 0:
        return min_bins
    b = int(n // target_per_bin)
    if b < min_bins:
        b = min_bins
    if b > max_bins:
        b = max_bins
    return b


def uniformity_metrics_per_chrom_scipy(
    data0_per_chrom,
    bins="auto",
    min_snps=10,
    target_per_bin=8,
    min_bins=20,
    max_bins=200,
    ):
    """
    data0_per_chrom: list of np.ndarray (datos_filtrados[0]) with SNP positions per chromosome
    bins: auto or int 
    min_snps: minimum SNPs to compute metrics skip/NaN metrics if fewer SNPs (too little data)
    target_per_bin, min_bins, max_bins: parameters for bins="auto"
    """
    results = []

    for c, pos in enumerate(data0_per_chrom):
        chrom_id = c + 1
        pos = np.asarray(pos, dtype=np.int64)

        if pos.size < min_snps:
            results.append({
                "chrom": chrom_id,
                "n_snps": int(pos.size),
                "range_bp": 0,
                "bins_used": np.nan,
                "ks_stat": np.nan, "ks_p": np.nan,
                "cvm_stat": np.nan, "cvm_p": np.nan,
                "chi2_stat": np.nan, "chi2_p": np.nan,
                "cv_counts": np.nan,
            })
            continue

        pos = np.unique(pos)
        pos.sort()
        pos_min = int(pos[0])
        pos_max = int(pos[-1])
        rango = pos_max - pos_min

        if rango <= 0:
            # All SNPs at same position (degenerate)
            results.append({
                "chrom": chrom_id,
                "n_snps": int(pos.size),
                "range_bp": int(rango),
                "bins_used": np.nan,
                "ks_stat": 1.0, "ks_p": 0.0,
                "cvm_stat": np.nan, "cvm_p": np.nan,
                "chi2_stat": np.nan, "chi2_p": np.nan,
                "cv_counts": np.nan,
            })
            continue

        # Normalize to [0,1]
        x = (pos - pos_min) / float(rango)

        # --- KS vs Uniform(0,1) ---
        ks = kstest(x, "uniform", args=(0.0, 1.0))
        ks_stat = float(ks.statistic)
        ks_p = float(ks.pvalue)

        # --- Cramér–von Mises vs Uniform(0,1) ---
        cvm = cramervonmises(x, "uniform", args=(0.0, 1.0))
        cvm_stat = float(cvm.statistic)
        cvm_p = float(cvm.pvalue)

         # --- Binned metrics (chi², CV) ---
        if bins == "auto":
            bins_c = _choose_bins(
                pos.size,
                target_per_bin=target_per_bin,
                min_bins=min_bins,
                max_bins=max_bins,
            )
        else:
            bins_c = int(bins)

        counts, _ = np.histogram(pos, bins=bins_c, range=(pos_min, pos_max + 1))
        expected = np.full_like(counts, fill_value=counts.sum() / counts.size, dtype=float)

        # chisquare expects expected sums equal observed sums; this holds by construction
        # note: interpret p-value cautiously
        chi2 = chisquare(counts, f_exp=expected)
        chi2_stat = float(chi2.statistic)
        chi2_p = float(chi2.pvalue)

        mean_counts = counts.mean()
        cv_counts = float(counts.std(ddof=0) / mean_counts) if mean_counts > 0 else np.nan

        results.append({
            "chrom": chrom_id,
            "n_snps": int(pos.size),
            "range_bp": int(rango),
            "bins_used": int(bins_c),
            "ks_stat": ks_stat, "ks_p": ks_p,
            "cvm_stat": cvm_stat, "cvm_p": cvm_p,
            "chi2_stat": chi2_stat, "chi2_p": chi2_p,
            "cv_counts": cv_counts,
        })

    return results

#Usa solo el criterio pasado en key
def rank_chromosomes_1metric(metrics, key="ks_stat"):
    """
    metrics: list of dicts with at least 'chrom' and the metric 'key'
    Returns a list aligned by chrom (index = chrom-1),
    with added fields: 'rank' and 'uniformity_class'.
    Usage:
    metrics = uniformity_metrics_per_chrom_scipy(datos_filtrados[0], bins=100, min_snps=10)
    uniformity = rank_chromosomes(metrics, key="ks_stat")
    for c in range(numcroms):
        u = uniformity[c]

        chrom = u["chrom"]
        rank = u["rank"]
        cls  = u["uniformity_class"] # clasificación close, media o strong deviation

        out.write(
            f"{chrom}\t{rank}\t{cls}\n"
        )
    
    """
    # Ordenar por desviación (menor = más uniforme)
    ranked = sorted(
        metrics,
        key=lambda d: (np.nan_to_num(d[key], nan=np.inf), d["chrom"])
    )

    # Asignar rank (1 = más uniforme)
    for r, d in enumerate(ranked, start=1):
        d["_rank_tmp"] = r

    # Clasificación por terciles
    vals = np.array([d[key] for d in ranked])
    q1, q2 = np.nanpercentile(vals, [33, 66])

    for d in ranked:
        v = d[key]
        if v <= q1:
            d["_class_tmp"] = "close_to_uniform"
        elif v <= q2:
            d["_class_tmp"] = "medium_deviation"
        else:
            d["_class_tmp"] = "strong_deviation"

    # Reordenar por cromosoma y limpiar campos temporales
    by_chrom = sorted(ranked, key=lambda d: d["chrom"])

    for d in by_chrom:
        d["rank"] = d.pop("_rank_tmp")
        d["uniformity_class"] = d.pop("_class_tmp")

    return by_chrom

def _percentile_scores(values):
    """
    Map array to [0,1] using rank-based percentiles with tie-averaging.
    Smaller values -> smaller percentile scores.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n <= 1:
        return np.zeros(n, dtype=float)

    order = np.argsort(v)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)

    # Tie handling: average rank within equal-value blocks
    sorted_v = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg = 0.5 * (i + j)
        ranks[order[i:j + 1]] = avg
        i = j + 1

    return ranks / (n - 1)

def _finite_values(metrics, k):
    vals = []
    for dd in metrics:
        v = dd.get(k, np.nan)
        try:
            v = float(v)
        except Exception:
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals, dtype=float)

def _metric_thresholds(metrics, keys, q_good=25, q_bad=75):
    """
    Per-metric thresholds based on quantiles of the metric values.
    Returns dict: key -> (good_thr, bad_thr)
    """
    thr = {}
    for k in keys:
        vals = _finite_values(metrics, k)
        if vals.size == 0:
            continue
        thr[k] = (float(np.nanpercentile(vals, q_good)),
                  float(np.nanpercentile(vals, q_bad)))
    return thr

def _classify_value(v, good_thr, bad_thr):
    """Close if <=good_thr, strong if >=bad_thr, else medium."""
    if not np.isfinite(v):
        return "insufficient_data"
    if v <= good_thr:
        return "close_to_uniform"
    if v >= bad_thr:
        return "strong_deviation"
    return "medium_deviation"

def rank_chromosomes(
    metrics,
    key="ks_stat",
    class_mode="multivariate",
    q_good=25,
    q_bad=75,
    require=2,
    class_keys=None,
    composite_keys=None,
    composite_agg="median",
):
    """
    Rank and classify chromosomes by spatial uniformity.

    - If key != "composite": rank by metrics[key] (smaller = more uniform).
    - If key == "composite": rank by a composite score built from percentile ranks across composite_keys.
        * composite_score = median(percentile_scores) by default (robust, conservative)
        * uniformity_class is monotonic with composite_score (quantile-based)
        * also stores per-metric classes in metric_classes and percentile scores in metric_percentiles

    Returns: list aligned by 'chrom' (sorted by chrom), each dict augmented with:
      - 'rank' (1..numcroms)
      - 'uniformity_class'
      - if key=="composite": 'composite_score', 'metric_percentiles', 'metric_classes'

    Usage:

    metrics = uniformity_metrics_per_chrom_scipy(datos_filtrados[0], bins="auto")
    uniformity = rank_chromosomes(metrics, key="composite")  # mediana por defecto

    for c in range(numcroms):
        u = uniformity[c]
        rank = u["rank"]                   # 1..numcroms
        cls  = u["uniformity_class"]       # close/medium/strong (monotónica con composite_score)
        score = u["composite_score"]       # 0..1 (mediana de percentiles)
        per_metric = u["metric_classes"]   # dict: {"ks_stat": "...", "cvm_stat": "...", "cv_counts": "..."}
    

    """

    # Default keys
    if class_keys is None:
        candidate = ["ks_stat", "cvm_stat", "cv_counts"]
        class_keys = [k for k in candidate if any(k in dd for dd in metrics)]

    if composite_keys is None:
        composite_keys = list(class_keys)

    # Precompute per-metric thresholds for per-metric classes (on raw metric scale)
    metric_thr = _metric_thresholds(metrics, composite_keys, q_good=q_good, q_bad=q_bad)

    # --- Composite path ---
    if key == "composite":
        # 1) Compute percentile scores per metric, per chromosome
        # We'll build: chrom -> dict(metric -> percentile_score)
        chroms = [int(d["chrom"]) for d in metrics]
        pct_by_chrom = {c: {} for c in chroms}

        for k in composite_keys:
            # gather finite values + chroms
            vals = []
            chrs = []
            for d in metrics:
                v = d.get(k, np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                if np.isfinite(v):
                    vals.append(v)
                    chrs.append(int(d["chrom"]))

            if len(vals) < 2:
                continue

            pct = _percentile_scores(np.array(vals, dtype=float))  # 0..1

            for c, s in zip(chrs, pct):
                pct_by_chrom[c][k] = float(s)

        # 2) Aggregate percentiles into composite_score (median by default)
        comp_score = {}
        for c in chroms:
            scores = [pct_by_chrom[c].get(k, np.nan) for k in composite_keys]
            scores = np.array(scores, dtype=float)
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                comp_score[c] = np.nan
            else:
                if composite_agg == "mean":
                    comp_score[c] = float(np.mean(scores))
                else:
                    # default median
                    comp_score[c] = float(np.median(scores))

        # 3) Rank by composite_score
        ranked = sorted(
            chroms,
            key=lambda c: (np.nan_to_num(comp_score.get(c, np.nan), nan=np.inf), c)
        )
        rank_map = {c: r for r, c in enumerate(ranked, start=1)}

        # 4) Define uniformity_class *monotonic with composite_score*
        finite_scores = np.array([comp_score[c] for c in chroms if np.isfinite(comp_score[c])], dtype=float)
        class_map = {c: "insufficient_data" for c in chroms} #inicializa
        if finite_scores.size > 0:
            good_thr = float(np.nanpercentile(finite_scores, q_good))
            bad_thr = float(np.nanpercentile(finite_scores, q_bad))
            for c in chroms:
                v = comp_score.get(c, np.nan)
                class_map[c] = _classify_value(v, good_thr, bad_thr)

        # 5) Per-metric classes (based on raw metric thresholds per metric)
        metric_class_by_chrom = {c: {} for c in chroms}
        for c in chroms:
            # Need raw metric values (not percentiles) for interpretability
            # Find the metric dict for chrom c
            d = next(dd for dd in metrics if int(dd["chrom"]) == c)
            for k in composite_keys:
                if k not in metric_thr:
                    continue
                gthr, bthr = metric_thr[k]
                v = d.get(k, np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                metric_class_by_chrom[c][k] = _classify_value(v, gthr, bthr)

        # 6) Return list aligned by chromosome, with fields added
        by_chrom = sorted(metrics, key=lambda d: int(d["chrom"]))
        for d in by_chrom:
            c = int(d["chrom"])
            d["composite_score"] = float(comp_score.get(c, np.nan)) if np.isfinite(comp_score.get(c, np.nan)) else np.nan
            d["rank"] = int(rank_map.get(c, np.nan))
            d["uniformity_class"] = class_map.get(c, "insufficient_data")
            d["metric_percentiles"] = pct_by_chrom.get(c, {})
            d["metric_classes"] = metric_class_by_chrom.get(c, {})
        return by_chrom

    # --- Non-composite path (keeps previous behavior) ---
    ranked = sorted(
        metrics,
        key=lambda d: (
            np.nan_to_num(d.get(key, np.nan), nan=np.inf),
            int(d["chrom"])
        )
    )
    rank_map = {int(d["chrom"]): r for r, d in enumerate(ranked, start=1)}

    # Classification: keep your multivariate rule (2-of-3 etc.) for non-composite ranking
    class_map = {int(dd["chrom"]): "insufficient_data" for dd in metrics} #inicializa
    if class_mode == "terciles":
        vals = _finite_values(metrics, key)
        if vals.size > 0:
            q1, q2 = np.nanpercentile(vals, [33, 66])
            for dd in metrics:
                chrom = int(dd["chrom"])
                v = dd.get(key, np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                if not np.isfinite(v):
                    continue
                if v <= q1:
                    class_map[chrom] = "close_to_uniform"
                elif v <= q2:
                    class_map[chrom] = "medium_deviation"
                else:
                    class_map[chrom] = "strong_deviation"
    else:
        thr = _metric_thresholds(metrics, class_keys, q_good=q_good, q_bad=q_bad)
        for dd in metrics:
            chrom = int(dd["chrom"])
            good_hits = 0
            bad_hits = 0
            used = 0
            for k in class_keys:
                if k not in thr:
                    continue
                v = dd.get(k, np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                if not np.isfinite(v):
                    continue
                used += 1
                gthr, bthr = thr[k]
                if v <= gthr:
                    good_hits += 1
                if v >= bthr:
                    bad_hits += 1

            if used == 0:
                class_map[chrom] = "insufficient_data"
            elif good_hits >= require:
                class_map[chrom] = "close_to_uniform"
            elif bad_hits >= require:
                class_map[chrom] = "strong_deviation"
            else:
                class_map[chrom] = "medium_deviation"

    by_chrom = sorted(metrics, key=lambda d: int(d["chrom"]))
    for d in by_chrom:
        chrom = int(d["chrom"])
        d["rank"] = int(rank_map.get(chrom, np.nan))
        d["uniformity_class"] = class_map.get(chrom, "insufficient_data")
    return by_chrom

def get_metrics(metric_classes,class_type="strong_deviation", order=None, exclude = False):
    """
    Return a list of metric names whose class matches `class_type`.
    or (if exclude=True) whose class is different from `class_type`.

    Parameters
    ----------
    metric_classes : dict
        Mapping metric name -> class label
    class_type : str
        Class to select (e.g. "strong_deviation", "close_to_uniform")
    order : list[str] or None
        Optional order for the output list

    Returns
    -------
    list[str]
        Metric names matching the requested class
    """
    
    if not metric_classes:
        return []
    if exclude:
        metrics = [
            name for name, cls in metric_classes.items()
            if cls != class_type
        ]
    else:
        metrics = [
            name for name, cls in metric_classes.items()
            if cls == class_type
        ]

    if order is not None:
        metrics = [k for k in order if k in metrics]

    return metrics

'''
End of v0.3 with SUMR added
'''


'''
***********************************************
v0.3 December 2025
RANDOM
False positive testing for HGsI
Functions for generating simulated data under specific conditions 
***********************************************
'''

def _beta_params(mode, strength=3.0):
    """
    Return (a,b) parameters for a Beta(a,b) on [0,1].

    strength controls clustering:
      - for center: larger strength -> more concentrated near 0.5 (a=b=strength)
      - for extremes: smaller "alpha" -> more mass near 0 and 1 (a=b=alpha<1)
      - for left/right: strong skew via (a<1, b>1) or vice versa
    """
    if mode == "uniform":
        return None  # handled separately

    if mode == "center":
        # a=b>1 concentrates around 0.5
        a = max(1.01, float(strength))
        return (a, a)

    if mode == "extremes":
        # a=b<1 concentrates near 0 and 1
        # interpret strength as "how extreme": larger strength => smaller alpha
        # you can tune this mapping; this one is stable and intuitive.
        alpha = 1.0 / (1.0 + float(strength))  # e.g. strength=3 -> alpha=0.25
        alpha = max(0.05, min(0.95, alpha))
        return (alpha, alpha)

    if mode == "left":
        # mass near 0: a<1, b>1
        alpha = 1.0 / (1.0 + float(strength))
        alpha = max(0.05, min(0.95, alpha))
        beta = 1.0 + float(strength)
        return (alpha, beta)

    if mode == "right":
        # mass near 1: a>1, b<1
        alpha = 1.0 / (1.0 + float(strength))
        alpha = max(0.05, min(0.95, alpha))
        beta = 1.0 + float(strength)
        return (beta, alpha)

    raise ValueError(f"Unknown mode: {mode!r}")


def _sample_unique_positions_in_range(pos_min, pos_max, n, rng, mode="uniform", strength=3.0):
    """
    Sample n unique integer positions in [pos_min, pos_max] (inclusive),
    with spatial pattern controlled by mode.

    Returns: np.ndarray sorted int64 of length n.
    """
    pos_min = int(pos_min)
    pos_max = int(pos_max)
    n = int(n)

    if n <= 0:
        return np.array([], dtype=np.int64)

    L = pos_max - pos_min + 1
    if L <= 0:
        raise ValueError("Invalid range: pos_max < pos_min")
    if n > L:
        raise ValueError(f"Cannot sample {n} unique positions from range length {L}")

    # Oversample factor: clustering can create many duplicates after flooring.
    # We'll iterate until enough uniques.
    oversample = max(2000, int(3.0 * n))
    got = np.empty(0, dtype=np.int64)

    params = _beta_params(mode, strength=strength)

    while got.size < n:
        if mode == "uniform":
            # simplest: sample without replacement directly
            # but only works efficiently if L not gigantic? It's fine: choice uses algorithmic sampling.
            new = rng.choice(L, size=min(oversample, L), replace=False).astype(np.int64)
            new = pos_min + new
        else:
            a, b = params
            u = rng.beta(a, b, size=oversample)
            # map to [pos_min, pos_max]
            new = pos_min + np.floor(u * (pos_max - pos_min + 1)).astype(np.int64)
            # clip safety
            new = np.clip(new, pos_min, pos_max)

        got = np.unique(np.concatenate([got, new]))

        # If clustering is too strong, we may progress slowly: increase oversample adaptively
        if got.size < n and oversample < 10_000_000:
            oversample = int(oversample * 1.5)

    # choose exactly n and sort
    # note: got is already sorted because np.unique sorts
    if got.size > n:
        # choose n from got without replacement, then sort
        sel = rng.choice(got.size, size=n, replace=False)
        out = np.sort(got[sel])
    else:
        out = got

    return out.astype(np.int64)


def generate_controlled_data0_perchr(
    data0_per_chrom,
    mode="uniform",
    strength=3.0,
    *,
    rng=None,
    per_chrom_mode=None
):
    """
    Generate a synthetic data0_per_chrom with the same (pos_min,pos_max,n_snps) per chromosome
    as the input, but with controlled spatial distribution.

    Parameters
    ----------
    data0_per_chrom : list[np.ndarray]
        Usually datos_filtrados[0], one array of SNP positions per chromosome.
    mode : str
        One of: "uniform", "extremes", "left", "right", "center".
        Ignored if per_chrom_mode is provided.
    strength : float
        Controls clustering intensity (meaning depends on mode).
    rng : np.random.Generator or None
        If None, uses np.random.default_rng().
    per_chrom_mode : list[str] or None
        Optional list of modes per chromosome (length = numcroms).
        Example: ["uniform","center",...]

    Usage
    ----------
    # Original data
    data0 = datos_filtrados[0]

    ss = np.random.SeedSequence(args.seed)
    rngs = ss.spawn(5)

    rng_uniform  = np.random.default_rng(rngs[0])
    rng_center   = np.random.default_rng(rngs[1])
    rng_left     = np.random.default_rng(rngs[2])
    rng_right    = np.random.default_rng(rngs[3])
    rng_extremes = np.random.default_rng(rngs[4])
   
    
    # (i) Uniforme
    data0_u = generate_controlled_data0(data0, mode="uniform", rng=rng_uniform)

    # (ii) Cluster a ambos extremos (más fuerte con strength mayor)
    data0_ext = generate_controlled_data0(data0, mode="extremes", strength=3.0, rng=rng_extremes)

    # (ii-b) Cluster solo a la izquierda o derecha
    data0_left  = generate_controlled_data0(data0, mode="left",  strength=3.0, rng=rng_left)
    data0_right = generate_controlled_data0(data0, mode="right", strength=3.0, rng=rng_right)

    # (iii) Cluster hacia el centro
    data0_center = generate_controlled_data0(data0, mode="center", strength=4.0, rng=rng_center)  

    """
    if rng is None:
        rng = np.random.default_rng()

    out = []
    numcroms = len(data0_per_chrom)

    if per_chrom_mode is not None:
        if len(per_chrom_mode) != numcroms:
            raise ValueError("per_chrom_mode must have the same length as data0_per_chrom")

    for c, pos in enumerate(data0_per_chrom):
        pos = np.asarray(pos, dtype=np.int64)

        if pos.size == 0:
            out.append(pos.copy())
            continue

        pos_min = int(pos.min())
        pos_max = int(pos.max())
        n = int(pos.size)

        m = per_chrom_mode[c] if per_chrom_mode is not None else mode

        syn = _sample_unique_positions_in_range(
            pos_min=pos_min,
            pos_max=pos_max,
            n=n,
            rng=rng,
            mode=m,
            strength=strength,
        )
        out.append(syn)

    return out

#version that assumes computations for all chromosomes without allowing specific ones
def generate_controlled_data0(data0_per_chrom, mode="uniform", strength=3.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    out = []
    for pos in data0_per_chrom:
        pos = np.asarray(pos, dtype=np.int64)
        if pos.size == 0:
            out.append(pos.copy())
            continue
        pos_min = int(pos.min())
        pos_max = int(pos.max())
        n = int(pos.size)
        syn = _sample_unique_positions_in_range(pos_min, pos_max, n, rng, mode=mode, strength=strength)
        out.append(syn)
    return out

# ---------- Candidate simulation under HG null ----------

def _pos_to_windows(pos, pos_min, W):
    # windows indexed 0..N-1
    return ((pos - pos_min) // W).astype(np.int64)

def _build_window_to_positions_map(pos_bp: np.ndarray, pos_min: int, W: int) -> dict:
    """
    Build a mapping window_id -> np.ndarray of SNP positions (bp) within that window.
    Only windows that contain at least one SNP in pos_bp are included.
    """
    pos_bp = np.asarray(pos_bp, dtype=np.int64)
    if pos_bp.size == 0:
        return {}

    win = ((pos_bp - int(pos_min)) // int(W)).astype(np.int64)

    # Sort by window to group efficiently
    order = np.argsort(win, kind="mergesort")
    win_sorted = win[order]
    pos_sorted = pos_bp[order]

    # Find group boundaries
    uniq, start_idx = np.unique(win_sorted, return_index=True)
    end_idx = np.r_[start_idx[1:], pos_sorted.size]

    m = {}
    for w, a, b in zip(uniq, start_idx, end_idx):
        m[int(w)] = pos_sorted[a:b]  # already sorted bp positions for that window
    return m


def _windows_to_random_snp_positions(win_ids: np.ndarray, win_to_pos: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Convert an array of window ids to an array of SNP bp positions by sampling
    one SNP uniformly at random from the SNPs present in each window.

    Assumes win_ids are unique (true if sampled without replacement).
    """
    win_ids = np.asarray(win_ids, dtype=np.int64)
    if win_ids.size == 0:
        return win_ids.astype(np.int64)

    out = np.empty(win_ids.size, dtype=np.int64)
    for i, w in enumerate(win_ids):
        arr = win_to_pos.get(int(w))
        if arr is None or arr.size == 0:
            # Should not happen if universes and candidates are derived from data0 windows
            # but keep a defensive fallback.
            out[i] = np.int64(-1)
        else:
            out[i] = arr[rng.integers(0, arr.size)]
    # Remove any defensive -1 (shouldn't exist); sort for aesthetics
    out = out[out >= 0]
    out.sort()
    return out

def _make_universe_windows_from_data0(data0_per_chrom, W):
    """
    For each chromosome return:
      - pos_min
      - universe U as np.ndarray of window ids (int64), sorted unique
    If W==1, window ids coincide with SNP offsets (0..Nc-1) only if SNPs are dense;
    for HGkI we actually want universe = windows occupied by data0 SNPs when W>1,
    and for W==1 universe = SNP positions themselves (exact-match universe).
    """
    universes = []
    for pos in data0_per_chrom:
        pos = np.asarray(pos, dtype=np.int64)
        if pos.size == 0:
            universes.append((0, np.array([], dtype=np.int64)))
            continue
        pos_min = int(pos.min())
        if W == 1:
            # exact universe = SNP positions (bp coordinates)
            U = np.unique(pos)
        else:
            U = np.unique(_pos_to_windows(pos, pos_min, W))
        universes.append((pos_min, U.astype(np.int64)))
    return universes


def _sample_candidates_from_universe(universes, n_i_per_chrom, rng):
    """
    universes: list of (pos_min, U) with U array of window IDs or SNP positions (if W==1)
    n_i_per_chrom: list[int] per chromosome, how many candidates for this method in each chromosome
    returns list[np.ndarray] per chromosome of sampled candidates (same domain as U)
    """
    out = []
    for (pos_min, U), n in zip(universes, n_i_per_chrom):
        n = int(n)
        if n <= 0 or U.size == 0:
            out.append(np.array([], dtype=np.int64))
            continue
        if n > U.size:
            # cannot sample without replacement; cap (or raise). Here: cap, but you may prefer error.
            n = int(U.size)
        sel = rng.choice(U.size, size=n, replace=False)
        out.append(np.sort(U[sel]).astype(np.int64))
    return out


# ---------- Main estimator ----------

def estimate_fp_rate_hgki(
    datos_filtrados_real,
    n_list_per_method,
    W=1,
    scenario="uniform",
    strength=3.0,
    R=20,
    S=50,
    alpha=0.05,
    decision="per_chrom",
    seed=123,
    ss_scenario=None,
    reorder_by_size=True,
    hg_test_func=None,
    hg_test_kwargs=None,
):
    """
    Estimate false-positive rate of HGkI under controlled SNP spatial scenarios.

    Parameters
    ----------
    datos_filtrados_real : list[list[np.ndarray]]
        Your real datos_filtrados structure. We use only datos_filtrados_real[0] as template.
    n_list_per_method : list[list[int]]
        For each method i (k methods), a list of length numcroms with candidate counts per chromosome.
        Example for k methods: [n1_per_chrom, n2_per_chrom, ..., nk_per_chrom]
    W : int
        Window size used by HGkI. W=1 means exact SNP matches.
    scenario : str
        "uniform", "center", "extremes", "left", "right"
    strength : float
        Controls clustering intensity.
    R, S : int
        Number of synthetic universes (R) and resamples per universe (S).
    alpha : float
        Significance threshold.
    decision : str
        "per_chrom"  -> FP across all chromosome-level tests (R*S*numcroms)
        "any_chrom"  -> FP per replicate: whether ANY chromosome is significant (R*S)
    seed : int
        Base seed.
    reorder_by_size : bool
        Passed to your HGkI implementation if applicable.
    hg_test_func : callable
        Your function that runs HGkI. Must accept at least:
          hg_test_func(datos_filtrados=..., D=W, chrom_index=None, method_indices=..., reorder_by_size=...)
        If None, you must plug it in.
    hg_test_kwargs : dict or None
        Extra kwargs passed to hg_test_func.

    Returns
    -------
    dict with:
      - fp_rate
      - fp_count
      - denom
      - fp_rate_per_universe (list length R)
      - fp_rate_per_chrom (list length numcroms)
      - details (scenario, strength, W, alpha, R, S, decision, k_methods)

    Usage
    -------
    Given in the main (RANDOM section)
    
    ss = np.random.SeedSequence(args.seed)
    ss_scenarios = ss.spawn(5)

    ss_uniform  = ss_scenarios[0]
    ss_center   = ss_scenarios[1]
    ss_left     = ss_scenarios[2]
    ss_right    = ss_scenarios[3]
    ss_extremes = ss_scenarios[4]

    for the uniform scenario:

    res_u = estimate_fp_rate_hgki(
        datos_filtrados_real=datos_filtrados,
        n_list_per_method=n_list_per_method,
        W=W,
        scenario="uniform",
        strength=strength,
        R=R,
        S=S,
        alpha=ALFA,
        decision="per_chrom",
        ss_scenario=ss_uniform,
        hg_test_func=hypergeom_kway_windows_test,
    )

    """
    if hg_test_func is None:
        raise ValueError("Provide your HGkI runner as hg_test_func=...")

    if hg_test_kwargs is None:
        hg_test_kwargs = {}

    data0_template = datos_filtrados_real[0]
    numcroms = len(data0_template)
    k = len(n_list_per_method)
    method_indices = list(range(1, k + 1))

    # sanity
    for i, nvec in enumerate(n_list_per_method):
        if len(nvec) != numcroms:
            raise ValueError(f"n_list_per_method[{i}] length {len(nvec)} != numcroms {numcroms}")

    # RNG hierarchy:
    # - If ss_scenario is provided (recommended), derive universes from it (ensures independence across scenarios).
    # - Otherwise fall back to seed (backward compatible).
    if ss_scenario is not None:
        # Accept either a SeedSequence or an integer seed
        if isinstance(ss_scenario, np.random.SeedSequence):
            ss_base = ss_scenario
        else:
            ss_base = np.random.SeedSequence(int(ss_scenario))
    else:
        ss_base = np.random.SeedSequence(int(seed))

    # --- Scenario normalization (empirical) ---
    scenario = str(scenario).lower()
    if scenario == "empirical":
        strength = 1.0 #ya está definido en el mapa pero por si acaso
        # Use the real SNP universe exactly as given; collapse all universes into a single one
        R0, S0 = int(R), int(S)
        R = 1
        S = R0 * S0

    ss_univ = ss_base.spawn(R)

    fp_rate_per_universe = []
    fp_count_total = 0
    denom_total = 0

    # For decision == "any_chrom": track how many chromosomes are significant per replicate
    k_sig_total = 0.0
    k_sig2_total = 0.0
    denom_k_total = 0

    k_sig_mean_per_universe = []
    k_sig_var_per_universe = []

    if decision == "per_chrom":
        fp_count_per_chrom = np.zeros(numcroms, dtype=int)  # global
        denom_per_chrom    = np.zeros(numcroms, dtype=int)  # global

    for r in range(R):
        rng_univ = np.random.default_rng(ss_univ[r])

        fp_count_r = 0
        denom_r = 0
        if decision == "per_chrom":
            fp_count_r_chrom = np.zeros(numcroms, dtype=int)  # por universo
            denom_r_chrom    = np.zeros(numcroms, dtype=int)

         # For any_chrom: per-universe moments of k_sig (number of significant chromosomes per replicate)
        elif decision == "any_chrom":
            k_sig_sum_r = 0.0
            k_sig2_sum_r = 0.0

        if scenario =="empirical":
            data0_syn =data0_template
        else:
            # generate controlled data0
            data0_syn = generate_controlled_data0(
                data0_template,
                mode=scenario,
                strength=strength,
                rng=rng_univ
            )

        # build universes (windows or exact SNP positions)
        universes = _make_universe_windows_from_data0(data0_syn, W)
        # For W>1, hypergeom candidates are sampled as window IDs, but HGkI expects bp positions.
        # Precompute, per chromosome, a mapping window_id -> SNP bp positions present in that window.
        if W > 1:
            win_to_pos_maps = []
            for c in range(numcroms):
                pos_c = np.asarray(data0_syn[c], dtype=np.int64)
                if pos_c.size == 0:
                    win_to_pos_maps.append({})
                    continue
                pos_min_c = int(pos_c.min())
                win_to_pos_maps.append(_build_window_to_positions_map(pos_c, pos_min_c, W))
        else:
            win_to_pos_maps = None

        # spawn S independent candidate-resampling RNGs for this universe
        ss_rep = ss_univ[r].spawn(S)
        
        for s in range(S):
            rng_rep = np.random.default_rng(ss_rep[s])

            # build synthetic candidate lists for k methods
            # datos_filtrados expected format: [data0, method1, method2, ...]
            syn = [data0_syn]

            for i in range(k):
                cand_i = _sample_candidates_from_universe(universes, n_list_per_method[i], rng_rep)

                if W == 1:
                    # Universe is exact SNP positions (bp) and candidates are bp.
                    syn.append(cand_i)
                else:
                    # Universe is windows, but HGkI expects bp positions to map -> windows internally.
                    # Convert each sampled window id into an actual SNP bp position present in that window.
                    cand_bp = []
                    for c in range(numcroms):
                        win_ids = np.asarray(cand_i[c], dtype=np.int64)
                        if win_ids.size == 0:
                            cand_bp.append(np.array([], dtype=np.int64))
                            continue
                        m = win_to_pos_maps[c]
                        bp = _windows_to_random_snp_positions(win_ids, m, rng_rep)
                        cand_bp.append(bp)
                    syn.append(cand_bp)

            

            res = hg_test_func(
                datos_filtrados=syn,
                D=W,  # keep your name D if your function uses it for W
                chrom_index=None,
                method_indices=method_indices,
                reorder_by_size=reorder_by_size,
                **hg_test_kwargs
            )

            # decision
            if decision == "per_chrom":
                for row in res["per_chrom"]:
                    c = row["chrom"] - 1  # chrom es 1-based
                    denom_per_chrom[c] += 1
                    denom_r_chrom[c] += 1
                    if float(row["p_value"]) <= alpha:
                        fp_count_per_chrom[c] += 1
                        fp_count_r_chrom[c] += 1

            elif decision == "any_chrom":
                # number of significant chromosomes in this replicate
                k_sig = sum(float(row["p_value"]) <= alpha for row in res["per_chrom"])
                if k_sig > 0:
                    fp_count_r += 1
                denom_r += 1

                # accumulate k_sig moments
                k_sig_sum_r += k_sig
                k_sig2_sum_r += (k_sig * k_sig)

            else:
                raise ValueError("decision must be 'per_chrom' or 'any_chrom'")

        if decision == "any_chrom":
            fp_rate_per_universe.append(fp_count_r / denom_r if denom_r > 0 else np.nan)
            fp_count_total += fp_count_r
            denom_total += denom_r
            # per-universe mean/var of k_sig
            if denom_r > 0:
                mean_k_r = k_sig_sum_r / denom_r
                # unbiased-ish variance across replicates within universe (optional ddof=1)
                # Here keep population variance; it's fine for summary
                var_k_r = (k_sig2_sum_r / denom_r) - (mean_k_r * mean_k_r)
            else:
                mean_k_r = np.nan
                var_k_r = np.nan

            k_sig_mean_per_universe.append(mean_k_r)
            k_sig_var_per_universe.append(var_k_r)

            # global accumulate across universes
            k_sig_total += k_sig_sum_r
            k_sig2_total += k_sig2_sum_r
            denom_k_total += denom_r

        elif decision== "per_chrom":
            fp_r = int(fp_count_r_chrom.sum())
            dn_r = int(denom_r_chrom.sum())
            fp_rate_per_universe.append(fp_r / dn_r if dn_r else np.nan)
            

    if decision == "per_chrom":
        fp_rate_per_chrom = np.divide( # division vectorizada, devuelve np.nan si denom es 0
            fp_count_per_chrom,
            denom_per_chrom,
            out=np.full_like(fp_count_per_chrom, np.nan, dtype=float),
            where=denom_per_chrom > 0
        )

        fp_count_total = int(fp_count_per_chrom.sum())
        denom_total    = int(denom_per_chrom.sum())
        fp_rate        = fp_count_total / denom_total if denom_total > 0 else np.nan

        fp_rate_per_chrom_out = fp_rate_per_chrom.tolist()
        mean_k_sig = None
        var_k_sig = None
    else:
        # decision == "any_chrom": ya está acumulado dentro del bucle de universos
        fp_rate = fp_count_total / denom_total if denom_total > 0 else np.nan
        fp_rate_per_chrom_out = None
        mean_k_sig = (k_sig_total / denom_k_total) if denom_k_total > 0 else np.nan
        var_k_sig  = (k_sig2_total / denom_k_total - mean_k_sig * mean_k_sig) if denom_k_total > 0 else np.nan   

    return {
        "fp_rate": fp_rate,
        "fp_count": int(fp_count_total),
        "denom": int(denom_total),
        "fp_rate_per_universe": fp_rate_per_universe,
        "fp_rate_per_chrom": fp_rate_per_chrom_out,
        "mean_k_sig_per_rep": mean_k_sig,
        "var_k_sig_per_rep": var_k_sig,
        "k_sig_mean_per_universe": k_sig_mean_per_universe,
        "k_sig_var_per_universe": k_sig_var_per_universe,

        "details": {
            "scenario": scenario,
            "strength": float(strength),
            "W": int(W),
            "alpha": float(alpha),
            "R_requested": int(R0) if scenario=="empirical" else int(R),
            "S_requested": int(S0) if scenario=="empirical" else int(S),
            "R": int(R),
            "S": int(S),
            "decision": decision,
            "k_methods": int(k),
        }
    }

'''
v0.3: End of functions for false positive testing on HGsI
'''



'''
***********************************************
v0.3 December 2025
RANDOM
False positive testing for TKL
Functions for generating simulated data under specific conditions 
***********************************************
'''
def _sample_candidates_from_data0_counts(data0_syn, n_per_chrom, rng):
    """
    Sample candidate SNP bp positions per chromosome from data0_syn without replacement.
    data0_syn: list[np.ndarray] SNP bp positions per chrom
    n_per_chrom: list[int] candidates per chrom
    Returns: list[np.ndarray] candidates per chrom
    """
    out = []
    for c, n in enumerate(n_per_chrom):
        n = int(n)
        if n <= 0:
            out.append(np.array([], dtype=np.int64))
            continue
        U = np.asarray(data0_syn[c], dtype=np.int64)
        if U.size == 0:
            out.append(np.array([], dtype=np.int64))
            continue
        if n > U.size:
            # Cap to avoid crash; should be rare in real usage
            n = int(U.size)
        sel = rng.choice(U.size, size=n, replace=False)
        arr = np.sort(U[sel]).astype(np.int64)
        out.append(arr)
    return out

def _precompute_tkl_expected_for_universe(
    data0_syn,
    n_list_per_method,
    numrs_expected,
    rng_univ,
    PARALLEL=False,
    num_processes=MAXPROC,
    block_size=BLOCK_SIZE,
):
    """
    Precompute, per chromosome:
      - maxsites
      - avertot_distances (expected distance profile)
      - e_median
      - TEMP flag (whether we can store null replicas)
      - if not TEMP: store null summaries (kl_null, median_null) to make p-values fast for many S
    """
    numcroms = len(data0_syn)
    k = len(n_list_per_method)

    precomp = []
    for c in range(numcroms):

        # Determine if testable: at least two methods with candidates in this chrom
        counts_c = [int(n_list_per_method[i][c]) for i in range(k)]
        nonEmpty = sum(1 for x in counts_c if x > 0)
        if nonEmpty < 2:
            precomp.append({
                "chrom": c + 1,
                "testable": False,
                "TEMP": False,
                "maxsites": 0,
                "avertot": None,
                "e_median": np.nan,
                "kl_null": None,
                "median_null": None,
            })
            continue

        # maxsites depends on sizes of candidate sets
        # We need dummy arrays only for sizes -> use zeros of those sizes
        dummy = [np.zeros(counts_c[i], dtype=np.int64) for i in range(k)]
        maxsites = int(numdists(*dummy))
        if maxsites <= 0:
            precomp.append({
                "chrom": c + 1,
                "testable": False,
                "TEMP": False,
                "maxsites": 0,
                "avertot": None,
                "e_median": np.nan,
                "kl_null": None,
                "median_null": None,
            })
            continue

        # Memory decision (same philosophy as your main TKL test)
        TEMP = False
        forma_array = (int(numrs_expected), int(maxsites))
        mem_ok, err, reqram = procesar_bloque(forma_array)
        if not mem_ok:
            # Check if at least a single vector fits
            forma_vec = (int(maxsites),)
            mem2_ok, _, _ = procesar_bloque(forma_vec)
            if not mem2_ok:
                raise MemoryError(err)
            TEMP = True

        avertot = np.zeros(maxsites, dtype=float)

        if not TEMP:
            # Store null replicas summaries for fast per-replicate p-values:
            # We do NOT store full rtot_distances (heavy); we store only KL and median for each null replicate
            kl_null = np.zeros(int(numrs_expected), dtype=float)
            median_null = np.zeros(int(numrs_expected), dtype=float)

            # First pass: generate rtot, accumulate expected mean, but also keep each rtot temporarily to compute KL later
            # To avoid storing all rtot arrays, we do it in two passes:
            #   pass1: store all rtot (list) if memory ok; your mem_ok means it is ok.
            # Here we can store rtot_distances as list as in main code.
            rtot_distances = []

            for r in range(int(numrs_expected)):
                rmuestra = []
                Uc = np.asarray(data0_syn[c], dtype=np.int64)
                for i in range(k):
                    ni = counts_c[i]
                    if ni <= 0:
                        rmuestra.append(np.array([], dtype=np.int64))
                    else:
                        sel = rng_univ.choice(Uc.size, size=ni, replace=False)
                        rmuestra.append(np.sort(Uc[sel]).astype(np.int64))
                rtot = totdist(*rmuestra, end=maxsites)
                rtot_distances.append(rtot)

            avertot = np.mean(np.vstack(rtot_distances), axis=0)
            e_med = float(np.percentile(avertot, 50))

            # Now compute null KL & median vs expected (reused for every S replicate)
            for r in range(int(numrs_expected)):
                rtot = rtot_distances[r]
                kl_null[r] = RelEntr(rtot, avertot)
                median_null[r] = np.percentile(rtot, 50)

            precomp.append({
                "chrom": c + 1,
                "testable": True,
                "TEMP": False,
                "maxsites": maxsites,
                "avertot": avertot,
                "e_median": e_med,
                "kl_null": kl_null,
                "median_null": median_null,
                "reqram_gib": float(reqram) if reqram is not None else None,
            })

        else:
            # TEMP mode: do not store replicas; compute expected only
            # We keep it sequential with rng_univ for reproducibility.
            Uc = np.asarray(data0_syn[c], dtype=np.int64)
            for r in range(int(numrs_expected)):
                rmuestra = []
                for i in range(k):
                    ni = counts_c[i]
                    if ni <= 0:
                        rmuestra.append(np.array([], dtype=np.int64))
                    else:
                        sel = rng_univ.choice(Uc.size, size=ni, replace=False)
                        rmuestra.append(np.sort(Uc[sel]).astype(np.int64))
                avertot += totdist(*rmuestra, end=maxsites)

            avertot /= float(numrs_expected)
            e_med = float(np.percentile(avertot, 50))

            precomp.append({
                "chrom": c + 1,
                "testable": True,
                "TEMP": True,
                "maxsites": maxsites,
                "avertot": avertot,
                "e_median": e_med,
                "kl_null": None,
                "median_null": None,
                "reqram_gib": float(reqram) if reqram is not None else None,
            })

    return precomp

def _tkl_eval_one_replicate(
    data0_syn,
    n_list_per_method,
    precomp,
    numrs_p,
    alpha,
    rng_rep,
):
    """
    Build one synthetic replicate (sample candidates for each method) and compute per-chrom p-values.
    Uses precomp expected profiles. In TEMP mode, p-values are computed by on-the-fly null resampling.
    """
    numcroms = len(data0_syn)
    k = len(n_list_per_method)

    # sample candidates for k methods (each a list of per-chrom arrays)
    cand_methods = []
    for i in range(k):
        cand_methods.append(_sample_candidates_from_data0_counts(data0_syn, n_list_per_method[i], rng_rep))

    per_chrom = []
    for c in range(numcroms):
        pc = precomp[c]
        if not pc["testable"]:
            per_chrom.append({
                "chrom": c + 1,
                "p_value": 1.0,
                "KL": 0.0,
                "o_median": np.nan,
                "e_median": pc["e_median"],
                "maxsites": 0,
                "TEMP": pc["TEMP"],
                "nonEmpty": 0,
            })
            continue

        maxsites = int(pc["maxsites"])
        avertot = pc["avertot"]
        e_med = float(pc["e_median"])
        TEMP = bool(pc["TEMP"])

        data_per_crom = [np.asarray(cand_methods[i][c], dtype=np.int64) for i in range(k)]
        nonEmpty = sum(1 for arr in data_per_crom if arr.size > 0)

        if nonEmpty < 2 or maxsites <= 0:
            per_chrom.append({
                "chrom": c + 1,
                "p_value": 1.0,
                "KL": 0.0,
                "o_median": np.nan,
                "e_median": e_med,
                "maxsites": int(maxsites),
                "TEMP": TEMP,
                "nonEmpty": int(nonEmpty),
            })
            continue

        tot_obs = totdist(*data_per_crom, end=maxsites)
        KL_obs = float(RelEntr(tot_obs, avertot))
        o_med = float(np.percentile(tot_obs, 50)) if tot_obs.size else np.nan

        if (not np.isfinite(o_med)) or (o_med > e_med):
            pval = 1.0
        else:
            if not TEMP:
                kl_null = pc["kl_null"]
                med_null = pc["median_null"]
                # Fast p-value using precomputed null summaries
                cond = (kl_null >= KL_obs) & (med_null <= e_med)
                b = int(np.sum(cond))
                R = int(cond.size)
                pval = (b + 1.0) / (R + 1.0) # v0.3 Davison–Hinkley correction
                #pval = float(np.mean(cond))
            else:
                # TEMP: on-the-fly null generation
                count = 0
                Uc = np.asarray(data0_syn[c], dtype=np.int64)
                counts_c = [int(n_list_per_method[i][c]) for i in range(k)]

                for _ in range(int(numrs_p)):
                    rmuestra = []
                    for i in range(k):
                        ni = counts_c[i]
                        if ni <= 0:
                            rmuestra.append(np.array([], dtype=np.int64))
                        else:
                            sel = rng_rep.choice(Uc.size, size=ni, replace=False)
                            rmuestra.append(np.sort(Uc[sel]).astype(np.int64))
                    rtot = totdist(*rmuestra, end=maxsites)
                    if (RelEntr(rtot, avertot) >= KL_obs) and (np.percentile(rtot, 50) <= e_med):
                        count += 1
                #pval = count / float(numrs_p) # v0.3 Davison–Hinkley correction pval = (count+1) / (float(numrs_p)+1)
                pval = (count+1.0) / (float(numrs_p)+1.0)

        per_chrom.append({
            "chrom": c + 1,
            "p_value": float(pval),
            "KL": float(KL_obs),
            "o_median": float(o_med),
            "e_median": float(e_med),
            "maxsites": int(maxsites),
            "TEMP": TEMP,
            "nonEmpty": int(nonEmpty),
        })

    return {"per_chrom": per_chrom}


def estimate_fp_rate_tkl(
    datos_filtrados_real,
    n_list_per_method,
    scenario="uniform",
    strength=3.0,
    R=1,
    S=100,
    numrs_expected=1000,
    numrs_p=None,
    alpha=0.05,
    decision="per_chrom",
    seed=123,
    ss_scenario=None,
    parallel=False,
    num_processes=MAXPROC,
    block_size=BLOCK_SIZE,
    verbose=False,
):
    """
    FP study for TKL distance test.

    - Builds synthetic SNP universe data0_syn per universe according to scenario.
    - Precomputes expected distance profiles once per universe (numrs_expected).
    - Runs S replicates by sampling candidates and computing p-values using the precomputed expected.

    decision:
      'per_chrom' -> FP across chromosome tests (R*S*numcroms)
      'any_chrom' -> FP per replicate: any chromosome significant (R*S)
    """

    def _vmsg(msg):
        
        if verbose:
            print(msg)
            logging.info(msg)

    if numrs_p is None:
        numrs_p = int(numrs_expected)

    data0_template = datos_filtrados_real[0]
    numcroms = len(data0_template)
    k = len(n_list_per_method)

    for i, nvec in enumerate(n_list_per_method):
        if len(nvec) != numcroms:
            raise ValueError(f"n_list_per_method[{i}] length {len(nvec)} != numcroms {numcroms}")

    # RNG hierarchy (same convention as HG)
    if ss_scenario is not None:
        if isinstance(ss_scenario, np.random.SeedSequence):
            ss_base = ss_scenario
        else:
            ss_base = np.random.SeedSequence(int(ss_scenario))
    else:
        ss_base = np.random.SeedSequence(int(seed))

    scenario = str(scenario).lower()

    # Empirical normalization: 1 universe, many replicates
    if scenario == "empirical":
        strength = 1.0
        R0, S0 = int(R), int(S)
        R = 1
        S = R0 * S0

    ss_univ = ss_base.spawn(int(R))

    fp_rate_per_universe = []

    if decision == "per_chrom":
        fp_count_per_chrom = np.zeros(numcroms, dtype=int)
        denom_per_chrom    = np.zeros(numcroms, dtype=int)
    else:
        fp_count_total = 0
        denom_total = 0
        k_sig_mean_per_universe = []

    for r in range(int(R)):
        rng_univ = np.random.default_rng(ss_univ[r])

        # Build SNP universe for this universe
        if scenario == "empirical":
            data0_syn = data0_template
        else:
            data0_syn = generate_controlled_data0(
                data0_template,
                mode=scenario,
                strength=strength,
                rng=rng_univ
            )

        # --- progress: scenario start ---
        # (counts of SNPs in universe can be useful)
        tot_snps = sum(len(np.asarray(x)) for x in data0_syn)
        _vmsg(
            f"[TKL-FP] scenario={scenario} | universe r={r+1}/{R} | "
            f"numrs_expected={numrs_expected} | S={S} | total_snps={tot_snps}"
        )

        # Precompute expected once for this universe
        _vmsg(f"[TKL-FP] scenario={scenario} | r={r+1}/{R} | computing expected profiles...")

        # Precompute expected once for this universe
        precomp = _precompute_tkl_expected_for_universe(
            data0_syn,
            n_list_per_method,
            numrs_expected=int(numrs_expected),
            rng_univ=rng_univ,
            PARALLEL=parallel,
            num_processes=num_processes,
            block_size=block_size,
        )

        # --- progress: expected ready ---
        # how many chromosomes were testable, and how many TEMP
        testable = sum(1 for pc in precomp if pc.get("testable", False))
        temp_ct = sum(1 for pc in precomp if pc.get("testable", False) and pc.get("TEMP", False))
        _vmsg(
            f"[TKL-FP] scenario={scenario} | r={r+1}/{R} | expected DONE | "
            f"testable_chrom={testable}/{numcroms} | TEMP_chrom={temp_ct}"
        )

        # Spawn replicate RNGs
        ss_rep = ss_univ[r].spawn(int(S))

        if decision == "per_chrom":
            fp_r_chrom = np.zeros(numcroms, dtype=int)
            dn_r_chrom = np.zeros(numcroms, dtype=int)
        else:
            fp_r = 0
            dn_r = 0
            k_sig_sum_r = 0.0

        for s in range(int(S)):
            rng_rep = np.random.default_rng(ss_rep[s])

            # Optional: progress every 10 replicates (only if verbose)
            if verbose and ((s + 1) % 25 == 0 or (s + 1) == 1 or (s + 1) == int(S)):
                _vmsg(f"[TKL-FP] scenario={scenario} | r={r+1}/{R} | replicate {s+1}/{S}")

            res = _tkl_eval_one_replicate(
                data0_syn=data0_syn,
                n_list_per_method=n_list_per_method,
                precomp=precomp,
                numrs_p=int(numrs_p),
                alpha=float(alpha),
                rng_rep=rng_rep
            )

            if decision == "per_chrom":
                for row in res["per_chrom"]:
                    c = int(row["chrom"]) - 1
                    denom_per_chrom[c] += 1
                    dn_r_chrom[c] += 1
                    if float(row["p_value"]) <= alpha:
                        fp_count_per_chrom[c] += 1
                        fp_r_chrom[c] += 1

            elif decision == "any_chrom":
                k_sig = sum(float(row["p_value"]) <= alpha for row in res["per_chrom"])
                if k_sig > 0:
                    fp_r += 1
                dn_r += 1
                k_sig_sum_r += float(k_sig)

            else:
                raise ValueError("decision must be 'per_chrom' or 'any_chrom'")

        # per-universe rate
        if decision == "per_chrom":
            fp_rate_per_universe.append(fp_r_chrom.sum() / dn_r_chrom.sum() if dn_r_chrom.sum() else np.nan)
        else:
            fp_rate_per_universe.append(fp_r / dn_r if dn_r else np.nan)
            fp_count_total += fp_r
            denom_total += dn_r
            k_sig_mean_per_universe.append(k_sig_sum_r / dn_r if dn_r else np.nan)

    out = {
        "fp_rate_per_universe": fp_rate_per_universe,
        "details": {
            "scenario": scenario,
            "strength": float(strength),
            "alpha": float(alpha),
            "R": int(R),
            "S": int(S),
            "decision": decision,
            "k_methods": int(k),
            "numrs_expected": int(numrs_expected),
            "numrs_p": int(numrs_p),
        }
    }

    if decision == "per_chrom":
        fp_rate_per_chrom = np.divide(
            fp_count_per_chrom,
            denom_per_chrom,
            out=np.full_like(fp_count_per_chrom, np.nan, dtype=float),
            where=denom_per_chrom > 0
        )
        fp_count_total = int(fp_count_per_chrom.sum())
        denom_total    = int(denom_per_chrom.sum())
        fp_rate        = fp_count_total / denom_total if denom_total else np.nan

        out.update({
            "fp_rate": float(fp_rate),
            "fp_count": int(fp_count_total),
            "denom": int(denom_total),
            "fp_rate_per_chrom": fp_rate_per_chrom.tolist(),
        })
    else:
        fp_rate = fp_count_total / denom_total if denom_total else np.nan
        out.update({
            "fp_rate": float(fp_rate),
            "fp_count": int(fp_count_total),
            "denom": int(denom_total),
            "k_sig_mean_per_universe": k_sig_mean_per_universe,
        })

    return out

# =========================
# TKL FP SUMMARY LOGGER
# =========================

def fp_summary_logger_tkl(fp_logger, title, res_fp, SCENARIOS, *, decision, top_k=10, alpha=None, numcroms=None):
    """
    Summary logger grouped by scenario (same style as HGkI FP summary).
    """
    if not res_fp:
        fp_logger.info(f"{title}: no results.")
        return

    det0 = res_fp[0].get("details", {})

    fp_logger.info("")
    fp_logger.info("=" * 78)
    fp_logger.info(title)
    fp_logger.info("=" * 78)
    fp_logger.info(
        "Design: "
        f"alpha={det0.get('alpha','NA')} | R={det0.get('R','NA')} | S={det0.get('S','NA')} | "
        f"k_methods={det0.get('k_methods','NA')} | numrs_expected={det0.get('numrs_expected','NA')} | "
        f"numrs_p={det0.get('numrs_p','NA')} | decision={det0.get('decision','NA')}"
    )

    if decision == "any_chrom" and (alpha is not None) and (numcroms is not None):
        fwer_ind = 1 - (1 - float(alpha))**int(numcroms)
        fp_logger.info(
            f"Expected FWER under independent chromosome tests (alpha={alpha}, C={numcroms}): {fwer_ind:.6g}"
        )

    fp_logger.info("-" * 78)

    for sc, res in enumerate(res_fp):
        scen = SCENARIOS[sc]
        det = res.get("details", {})

        fp_logger.info("")
        fp_logger.info(f"Scenario: {scen} | strength={det.get('strength', 'NA')}")
        fp_logger.info("=" * 78)

        fp_logger.info(
            f"Global: fp_rate={res.get('fp_rate', np.nan):.6g} | fp_count={res.get('fp_count','NA')} | denom={res.get('denom','NA')}"
        )

        u = np.asarray(res.get("fp_rate_per_universe", []), dtype=float)
        mean_u = np.nanmean(u) if u.size else np.nan
        var_u  = np.nanvar(u, ddof=1) if np.sum(~np.isnan(u)) > 1 else np.nan
        sd_u   = np.sqrt(var_u) if np.isfinite(var_u) else np.nan
        n_eff  = int(np.sum(~np.isnan(u))) if u.size else 0
        se_u   = sd_u / np.sqrt(n_eff) if (n_eff > 0 and np.isfinite(sd_u)) else np.nan

        fp_logger.info(
            f"Across universes (R={len(u)}): mean={mean_u:.6g} | sd={sd_u:.6g} | var={var_u:.6g} | n_eff={n_eff} | se={se_u:.6g}"
        )

        if decision == "per_chrom":
            fp_ch = res.get("fp_rate_per_chrom", None)
            if fp_ch is None:
                fp_logger.info("Chromosome rates: not available (decision != 'per_chrom').")
                continue

            fp_ch = np.asarray(fp_ch, dtype=float)
            idx = np.where(np.isfinite(fp_ch))[0]
            order = idx[np.argsort(fp_ch[idx])[::-1]]
            top = min(int(top_k), len(order))
            top_pairs = [(int(i + 1), float(fp_ch[i])) for i in order[:top]]

            fp_logger.info(
                "Top chromosomes by fp_rate: "
                + ", ".join([f"chr{c}:{v:.6g}" for c, v in top_pairs])
            )

            finite = fp_ch[np.isfinite(fp_ch)]
            if finite.size:
                fp_logger.info(
                    f"fp_rate_per_chrom summary: min={np.min(finite):.6g} | median={np.median(finite):.6g} | max={np.max(finite):.6g}"
                )

        elif decision == "any_chrom":
            ksig = res.get("k_sig_mean_per_universe", None)
            if ksig is not None:
                ksig = np.asarray(ksig, dtype=float)
                fp_logger.info(
                    f"k_sig mean across universes: mean={np.nanmean(ksig):.6g} | sd={np.nanstd(ksig, ddof=1):.6g}"
                    if np.sum(np.isfinite(ksig)) > 1 else
                    f"k_sig mean across universes: mean={np.nanmean(ksig):.6g}"
                )

'''
v0.3: End of functions for false positive testing on TKL
'''




'''
End of v0.3 
'''



def main():

    global ALFA
    global numrs  # 
    global RANDOM
    global TEST
    global FIGURE
    global STATS
    global UMBRAL1
    global UMBRAL2
   
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Scans an unspecified number of files.')

    # Los diferentes nombres de archivos se guardan en una lista llamada args.archivos
    parser.add_argument('archivos', metavar='N', type=str, nargs='*',
                        help='names of the files to be analyzed')

    # Argumento con nombre para el path
    parser.add_argument('--path', type=str, default=None,
                        help='optional path for files')

    # Argumento con nombre para el nivel de significación
    parser.add_argument('--SL', type=float, default=0.05,
                        help='significance level, a value between 0 and 1')
    parser.add_argument('--Kmax', type=int, default=np.inf,
                        help='Kmax significant sites per chromosome, an integer between 0 and 1E6')
    # Argumento con nombre para la distancia en las interseccions y para el filtrado de clusters dentro de cromosomas si el número de sitios es grande (>=Kmax)
    parser.add_argument('--dist', type=int, default=-1,
                        help='distances, an integer between 0 and 1E8')

    # Argumento con nombre para el número de permutaciones
    parser.add_argument('--perm', type=int, default=10000,
                        help='permutations, an integer between 0 and 1E6')

    # Argumento con nombre para hacer un control aleatorio
    parser.add_argument('--rand', action='store_true', default=False,
                        help='control using randomly generated data based on the pattern of the entered files')

    # Argumento con nombre para muestrear desde una uniforme
    parser.add_argument('--uniform', action='store_true', default=False,
                        help='resampling assuming uniform distribution of snps')
    
    # Argumento con nombre para solo calcular intersecciones de todos los cromosomas (sin hacer test)
    parser.add_argument('--notest', action='store_true', default=False,
                        help=(
            "Compute SNP intersections only. No statistical tests (KL or HGkI)"
             "are performed."
                        )
    )

    # Argumento con nombre para no aplicar la regla strict: si paso el argumento --permissive pone la variable de dest que es strict a False

    parser.add_argument('--permissive', action='store_false', dest='strict',
                    help='avoid strict rule for redundancy')

    # Intersecciones solo para el cromosoma indicado
    parser.add_argument('--chr-id',
                        type=positive_int,  # Usar la función de validación
                        default=None,      # Valor por defecto es None
                        
                        help=(
                            "Chromosome identifier (1-based). Valid values range from 1 to the "
                            "total number of chromosomes. If not provided, SNP intersections for "
                            "the given distance and the HGkI test (if enabled) are computed for all "
                            "chromosomes. When specified, computations are restricted to the "
                            "selected chromosome."
                        )
    )

    # Argumento con nombre para pintar las distribucioens de distancias observada y esperada
##    parser.add_argument('--paint', action='store_true', default=False,
##                        help='compute observed and expected histogram')
    parser.add_argument('--paint', nargs='?', const='all', default=False,
                    help='compute observed and expected histogram. Use --paint for all chromosomes or --paint <numchr> for a specific chromosome.')

    parser.add_argument('--max-xvalue', type=float, default=None,
                        help='Fixed maximum value for the X axis in mega bases (e.g., 20 for 20Mb).')

    parser.add_argument('--max-yvalue', type=float, default=None,
                        help='Fixed maximum value for the Y axis (e.g., 0.2 for a maximum frequency of 0.2).')

    # Argumento con nombre para estadísticas de datos por pantalla
    parser.add_argument('--stats', action='store_true', default=False,
                        help='compute stats for the given files')

    # V0.3 nuevos argumentos añadidos
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="Window size (bp) for the hypergeometric k-way intersection test. "
         "If not provided, defaults to 1. "
         "Use --W 1 for exact SNP-level coincidence."
         "Very large values of --W may result in a very small number of windows (N_windows), making the test uninformative."
    )
    
    parser.add_argument(
        "--HG",
        action="store_true",
        help=(
            "Run the hypergeometric k-way intersection test instead of "
            "the distance-based KLinterSel analysis."
        )
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output, including detailed warnings and debug information."
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None, use system entropy)."
    )

    parser.add_argument(
        "--fp-R",
        type=int,
        default=1,
        help="Number of universes (R) for false-positive studies."
    )

    # Argumento con nombre para evitar calcular las intersecciones
    parser.add_argument('--skip-intersection', action='store_true',
                        help=(
            "Skip SNP intersection computation."
                        )
    )

    parser.add_argument(
        "--fp-S",
        type=int,
        default=100,
        help="Number of replicates per universe (S) for false-positive studies."
    )


    def warn_large_fp_rs(R, S, *, context="FP study"):
        RS = int(R) * int(S)

        if RS > FP_RS_HARD:
            msg = (
                f"{context}: R×S = {RS:,} is extremely large. "
                "This run may take hours or days and consume substantial resources."
            )
            logging.warning(msg)
            print("WARNING:", msg)

        elif RS > FP_RS_WARN:
            msg = (
                f"{context}: R×S = {RS:,} is large. "
                "Consider reducing R or S if runtime becomes an issue."
            )
            logging.warning(msg)
            print("WARNING:", msg)


    # Parsear los argumentos
    args = parser.parse_args()

    #v0.3
    intersec = not args.skip_intersection
    R_MAX = 1000
    S_MAX = 10_000

    if not (1 <= args.fp_R <= R_MAX):
        raise ValueError(
            f"--fp-R must be between 1 and {R_MAX} (got {args.fp_R})."
        )

    if not (1 <= args.fp_S <= S_MAX):
        raise ValueError(
            f"--fp-S must be between 1 and {S_MAX} (got {args.fp_S})."
        )


    warn_large_fp_rs(args.fp_R, args.fp_S, context="False-positive analysis")

    R_fp = args.fp_R
    S_fp = args.fp_S

    # generador aleatorio
    ss_global = np.random.SeedSequence(args.seed)
    

    # Validar el valor de SL
    if not 0 <= args.SL <= 1:
        raise ValueError("The significance level must be between 0 and 1.")
    ALFA=args.SL

    # Validar el valor de dist
    if not -1 <= args.dist <= int(1E8):
        raise ValueError("The distance must be an integer between 0 and 1E8.")

    distance=10000 #default value
    if args.dist>=0:
        distance=args.dist
       
    # Validar el valor de perm
    if not 0 <= args.perm <= int(1E6):
        raise ValueError("The number of permutations must be an integer between 0 and 1E6.")
    elif args.perm!=numrs: #numrs=0 de modo que si no se introduce el argumento o se introduce cualquier valor <>0  el valor se carga en numrs
        numrs = args.perm

    if args.rand==True:
        RANDOM=True

    if args.notest==True:
        TEST=False

    if args.paint:
        
        if not TEST:
            mensaje="If you specify both --notest and --paint, no plots will be generated. The --paint option requires the expected distribution computed during the test, which is skipped when --notest is used."
            logging.warning(mensaje)
            print(f"\nWarning: {mensaje}")
            print(f"Please refer to the {Path(handler.baseFilename).name} file for more details.")
            #sys.exit(0)
        else:
            FIGURE=True

    if args.stats==True:
        STATS=True

    # Si no se proporcionan suficientes argumentos de archivos pero sí una ruta, preguntar al usuario
    #Son necesarios al menos tres archivos: uno con los SNPs originales y dos con los resultados de dos métodos de detección de selección
    currentnumberoffiles=len(args.archivos)
    requirednumber=3
    if currentnumberoffiles<requirednumber:
        print(f'At least {requirednumber} files are required to run the program and only {currentnumberoffiles} were passed')
        num_archivos = requirednumber-currentnumberoffiles
        if not args.archivos:
            args.archivos = []
        for i in range(currentnumberoffiles,requirednumber):
            nombre_archivo = input(f"Enter the file name {i+1}: ")
            args.archivos.append(nombre_archivo)

    if not args.path:
        dirdatos='.' # datos en el mismo sitio que el ejecutable
    else:
        dirdatos= args.path

    nombre=[]
    datos_filtrados=[]
    maxsnps_perfile=['']

    ########### READ THE FILES
    for num,file in enumerate(args.archivos):

        nombre.append(file)
        #identify the extension of the snps file
        lname=nombre[-1].split(sep='.') #lista de elementos del nombre de fichero separados por punto
        #arr0name=''.join(lname[0:-1]) # une todo el nombre menos la extensión
        extname=lname[-1]
        ruta=os.path.join(dirdatos,nombre[-1])
        if extname=='csv':
            nombre[-1]=eliminar_comas_csv(ruta,',')
            if not nombre[-1]:
                print(f'Problem after removing commas from thousands in {file}')
                sys.exit(1)
        else:
            nombre[-1]=eliminar_comas(ruta)
            if not nombre[-1]:
                print(f'Problem after removing commas from thousands in {file}')
                sys.exit(1)

        ruta = os.path.join(dirdatos, nombre[-1]) 

        ''' Lo que guarda datos_filtrados es una lista de datos para cada fichero.
            Para cada fila de la lista se guarda una lista de claves donde cada clave contiene un np array de posiciones de SNPs. Cada clave es un número de cromosoma
            En resumen:
                La primera dimensión los distintos ficheros. La segunda dimensión los distintos cromosomas. La tercera los valores de SNPs para ese cromosoma en ese fichero.
        '''

        if extname=='csv':
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(FilterCsv(ruta)[0])
            else: #Si hay demasiados sitios significativos (>=Kmax se filtran por clusters de distancia. Por defecto Kmax vale inf y en este caso no se filtra.
                datf,maxsnps=FilterCsv(ruta,Kmax=args.Kmax, D=distance)
                datos_filtrados.append(datf)
                maxsnps_perfile.append(maxsnps)
        elif extname=='map':
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(filter_crompos(ruta, cromcol=0, poscol=3, header=False)[0]) #map file cols are chrom id genetic_pos physical_pos
            else:
                datf,maxsnps=filter_crompos(ruta, cromcol=0, poscol=3,Kmax=args.Kmax, D=distance, header=False)
                datos_filtrados.append(datf)
                maxsnps_perfile.append(maxsnps)
        elif extname=='tped':
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(filter_crompos(ruta, cromcol=0, poscol=3)[0])
            else:
                datf,maxsnps=filter_crompos(ruta, cromcol=0, poscol=3,Kmax=args.Kmax, D=distance)
                datos_filtrados.append(datf)
                maxsnps_perfile.append(maxsnps)
        elif extname=='txt' or extname=='tsv':
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(filter_crompos(ruta, cromcol=0, poscol=1)[0])
            else:
                datf,maxsnps=filter_crompos(ruta, cromcol=0, poscol=1,Kmax=args.Kmax, D=distance)
                datos_filtrados.append(datf)
                maxsnps_perfile.append(maxsnps)
        elif extname=='hapflk':
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(filter_crompos(ruta, cromcol=1, poscol=2)[0])
            else:
                datf,maxsnps=filter_crompos(ruta, cromcol=1, poscol=2,Kmax=args.Kmax, D=distance)
                datos_filtrados.append(datf)
                maxsnps_perfile.append(maxsnps)
        elif extname=='norm': #selscan .norm file with --crit-val type 1 or --crit-percent
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(filter_norm(ruta,cromid=1, poscol=1, critcol=-1, criter=1)[0])
            else:
                datf,maxsnps=filter_norm(ruta,cromid=1, poscol=1, critcol=-1, criter=1,Kmax=args.Kmax, D=distance)
                datos_filtrados.append(datf)
                maxsnps_perfile.append(maxsnps)

        print(f'\nDATA OBTAINED FROM FILE {nombre[-1]}\n')
         
    numfiles= len(nombre)

    samefiles=True
    for i in range(1,numfiles-1):
        if nombre[i].upper()!=nombre[i+1].upper():
            samefiles=False
            break

    if random_files: #En vez de usar los datos obtenidos genera datos aleatorios basados en la estructura de los datos
        for i in range(1,numfiles):
            datos_filtrados[i] = genrandomdata(datos_filtrados[0],datos_filtrados[i])
    else: # En el caso de no ser random_files

        if args.strict: # v0.2 Opción conservadora que elimina un set de candidatos si coincide con el de otro método (implicaría métodos muy similares o iguales)

            #datos_filtrados, eliminados,  = filtrar_datos(datos_filtrados, umbral=0.75)

            datos_filtrados, eliminados,  = filtrar_datos_B(datos_filtrados, umbral1=UMBRAL1, umbral2=UMBRAL2, min_snps=5)

            numfiles= len(datos_filtrados) # actualiza el número total de ficheros

            currentnumberoffiles=numfiles

            if eliminados: #actualiza la lista de nombres


                for superset_idx, subset_list in eliminados.items():
                    nombre_superset = nombre[superset_idx]
                    for subset_idx in subset_list:
                        nombre_subset = nombre[subset_idx]
                        msg=f"File {nombre_subset} eliminated because at least {UMBRAL2:.0%} of its candidates are included in file {nombre_superset}.\n************************\n"
                        print('\n'+msg)
                        logging.warning(msg)
                #Actualiza la lista de nombres y de max snps per file
                indices_finales = set().union(*eliminados.values())
                nombre = [n for idx, n in enumerate(nombre) if idx not in indices_finales]
                maxsnps_perfile = [n for idx, n in enumerate(maxsnps_perfile) if idx not in indices_finales]

            if currentnumberoffiles<requirednumber:
                print(f'Only one candidate file remains after deleting identical or highly overlapping files with the non-permissive option. Use --permissive to prevent the deletion of candidate files.')
                sys.exit(0)

    #v0.3 Nombres de los ficheros sin extensión
     #lista de elementos del nombre de fichero separados por punto
    names_no_ext = [Path(n).stem for n in nombre] # une todo el nombre menos la extensión

    # Listas para cada fichero (posición 0 es el fichero total, las otras las de los métodos)
    #cada elemento de la lista contiene una lista con el número de sitios totales (primera lista) o significativos por cromosoma
    Lsitios=[]
    for data in datos_filtrados: # Cada elemento de datos filtrados corresponde a los datos de un fichero
        Lsitios.append([fila.size for fila in data]) #Lsitos tiene tantas filas como ficheros y tantas columnas como cromosomas. Los valores de celdas (dim 3) son el número de snps en ese cromosoma (columna dim 2) y fichero (fila dim 1)

    numcroms=len(Lsitios[0])
    #v0.3 the test for all or for a specific chromosome
    chrom_index = None if args.chr_id is None else (args.chr_id - 1)
    eff_numcroms= numcroms if args.chr_id is None else 1 #v0.3 

    print(f'The number of chromosomes is {numcroms}\n')

    if STATS: #acr Nov 2025
        
        for i in range(numfiles):
            Range=[]
            print(f"\nSTATS for FILE {nombre[i]}\n")
            
            for c in range(numcroms):
                
                print(f"CHROMOSOME {c+1}")
                Min=np.min(datos_filtrados[i][c])
                print("Min:", Min)
                Max= np.max(datos_filtrados[i][c])
                print("Max:", Max)
                Range.append(Max-Min+1) # Tamaño desde el primer hasta el último SNP
                print("Range:", Range[-1])
                print("Mean:", round(np.mean(datos_filtrados[i][c]),2))
                print("SD:", round(np.std(datos_filtrados[i][c]),2))
                print(f'Median {round(np.percentile(datos_filtrados[i][c],50),2)}')
            Range=np.array(Range)
            aveRange = Range.mean()
            sdRange = Range.std()
            print(f'Mean range per chromosome {aveRange} +- {sdRange}')

        print("\nKLinterSel DONE")

        return

    tot_distances= [-1]*numcroms # distancias totales entre los significativos de  todos los métodos para cada cromosoma
    cd_test= [0]*numcroms
    cd= [0]*numcroms # lista que almacenará para cada cromosoma los diccionarios de sitios comunes
    kset= [0]*numcroms # lista que almacenará para cada cromosoma los sets de posiciones que aparecen en los k métodos

    if KILOBASES:

        for i in range(1,numfiles):

            print(f'\nConverting data in {nombre[i]} to kb:\n')

        for data in datos_filtrados:

            for c in range(numcroms):

                data[c]=data[c]/1000

    maxsnps_perfile[0]=np.sum(Lsitios[0])
    print(f'The total number of SNPs in the original data {nombre[0]} is {maxsnps_perfile[0]}\n')

    if not TEST and not intersec:
        msg = (
            "No tests selected and SNP intersection computation is disabled. "
            "Nothing to do. Exiting."
        )
        logging.warning(msg)
        print(msg)
        sys.exit(0)


    for i in range(1,numfiles):
        if numcroms != len(Lsitios[i]):
            logging.error(
                f"Method {i}: chromosome count mismatch "
                f"({len(Lsitios[i])} found, {numcroms} expected). "
                "All methods must provide the same chromosome indexing."
            )
            sys.exit(1)
        if args.verbose:
            print(f'The number of significant SNPs in {nombre[i]} before filtration was {maxsnps_perfile[i]} and after was {np.sum(Lsitios[i])}. If there were more than {args.Kmax} significant SNPs, those within clusters closer than {distance} nucleotides were filtered out.')


    ########
    ######## CÁLCULO DEL ESTADÍSTICO Y REMUESTREO 
    ########

    # CALCULO NÚMERO DE CORES-1 POR SI HAY PARALELIZACIÓN

    num_processes=max(1,psutil.cpu_count(logical=False)-1)

    if TEST:
        #v0.3
        arrP=np.zeros(numcroms)

        if args.HG: # HGkI test

            moment2 = datetime.now()
            moment2 = moment2.strftime("%d%m%y_%H%M%S")
            FIGURE=False

            # V0.3 Window size for hypergeometric test
            W = args.W if args.W is not None else 1

            out_name2=""
            test=  'HGkItest'
            formatted_window = f"{W:.1E}".replace('+', '').replace('E0', 'E') # to print x with 3 decimals: f'x:.3E'
            out_name2 = test+ '_W'+formatted_window
            if not args.strict:
                out_name2+='_Perm'
            out_name2+='_'+moment2
            if samefiles:    
                out_name2+='_SAMEFILES'
           
            if RANDOM: # v0.3 false positives study # SCENARIOS = ["uniform", "center", "left", "right", "extremes" "empirical"]

                logging.info(
                    f"Running FP study on hypergeometric k-way intersection test (W={W})."
                )
                print(f"Running FP study on hypergeometric k-way intersection test (W={W}).\n")

                if args.seed is None:
                    logging.info("Random seed not provided: using system entropy.")
                else:
                    logging.info(f"Random seed set to {args.seed}.")

                if args.chr_id is not None:
                    logging.warning(
                        "FP study ignores --chr_id and is always run on all chromosomes."
                )

                # --- FP logger: separado para el estudio de falsos positivos HGkI ---
                fp_logger = logging.getLogger("KLinterSel.FP")
                fp_logger.setLevel(logging.INFO)
                fp_logfile = f"{out_name2}_FP.log"
                fp_handler = logging.FileHandler(fp_logfile, mode="a", delay=True)
                fp_handler.setFormatter(formatter)

                # Evitar duplicados si se llama más de una vez
                if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == fp_handler.baseFilename
                        for h in fp_logger.handlers):
                    fp_logger.addHandler(fp_handler)
                    
                fp_logger.propagate = False
                
                logging.info(f"HGkI FP study enabled. Detailed log: {out_name2}_FP.log")
                fp_logger.info("Starting HGkI false-positive study.")
                
                ss_scenarios = ss_global.spawn(len(SCENARIOS))
                strength_map = {
                    "uniform": 1.0,
                    "center": 3.0,
                    "left": 5.0,
                    "right": 5.0,
                    "extremes": 5.0,
                    "empirical": 1.0,
                }
                # decision per_chrom
                res_fp=[]
                for sc, scen in enumerate(SCENARIOS):

                    if scen not in strength_map:
                        raise ValueError(f"No strength defined for scenario '{scen}'")

                    strength = float(strength_map[scen])

                    res = estimate_fp_rate_hgki(
                        datos_filtrados_real=datos_filtrados,
                        n_list_per_method=Lsitios[1:],
                        W=W,
                        scenario=scen, 
                        strength=strength,
                        R=R_fp,
                        S=S_fp,
                        alpha=ALFA,
                        decision="per_chrom",
                        ss_scenario= ss_scenarios[sc],
                        hg_test_func=hypergeom_kway_windows_test,
                    )

                    res_fp.append(res)
                    
                # --- Global FP study header ---
                universe_file = names_no_ext[0]
                candidate_files = names_no_ext[1:]
                det0 = res_fp[0].get("details", {})  # todos comparten los mismos details

                fp_logger.info("=" * 78)
                fp_logger.info("HGkI FALSE-POSITIVE STUDY (decision=per_chrom)")
                fp_logger.info("=" * 78)

                fp_logger.info(f"Universe file (data0): {universe_file}")
                fp_logger.info(
                    "Candidate files (methods): "
                    + ", ".join(f"{f}" for f in candidate_files)
                )

                fp_logger.info(
                    "Candidate counts (sum per method): "
                    + ", ".join(
                        f"{s}={sum(nvec)}"
                        for s, nvec in zip(candidate_files, Lsitios[1:])
                    )
                )
                
                fp_logger.info(
                    "Design: "
                    f"W={det0.get('W', 'NA')} | alpha={det0.get('alpha', 'NA')} | "
                    f"R={det0.get('R', 'NA')} | S={det0.get('S', 'NA')} | "
                    f"k_methods={det0.get('k_methods', 'NA')} | "
                    f"decision={det0.get('decision', 'NA')}"
                )

                fp_logger.info("-" * 78)

                # --- Summary grouped by scenario ---
                for sc, res in enumerate(res_fp):
                    scen = SCENARIOS[sc]
                    det = res.get("details", {})
                    
                    fp_logger.info("")
                    fp_logger.info(f"Scenario: {scen} | strength={det.get('strength', 'NA')}")
                    fp_logger.info("=" * 78)

                    # 0) Global (scenario-specific)
                    fp_logger.info(
                        f"Global: fp_rate={res['fp_rate']:.6g} | fp_count={res['fp_count']} | denom={res['denom']}"
                    )

                    # 1) stats across universes
                    u = np.asarray(res["fp_rate_per_universe"], dtype=float)  # length R
                    mean_u = np.nanmean(u)
                    var_u  = np.nanvar(u, ddof=1) if np.sum(~np.isnan(u)) > 1 else np.nan
                    sd_u   = np.sqrt(var_u) if np.isfinite(var_u) else np.nan

                    n_eff = int(np.sum(~np.isnan(u)))
                    se_u = sd_u / np.sqrt(n_eff) if (n_eff > 0 and np.isfinite(sd_u)) else np.nan

                    fp_logger.info(
                        f"Across universes (R={len(u)}): mean={mean_u:.6g} | sd={sd_u:.6g} | var={var_u:.6g} | n_eff={n_eff} | se={se_u:.6g}"
                    )

                    # 2) chromosome-level info (only if decision == "per_chrom")
                    fp_ch = res.get("fp_rate_per_chrom", None)
                    if fp_ch is None:
                        fp_logger.info("Chromosome rates: not available (decision != 'per_chrom').")
                        continue

                    fp_ch = np.asarray(fp_ch, dtype=float)

                    # ranking (highest to lowest)
                    idx = np.where(np.isfinite(fp_ch))[0]
                    order = idx[np.argsort(fp_ch[idx])[::-1]]  # descending
                    top_k = min(10, len(order))
                    top_pairs = [(int(i + 1), float(fp_ch[i])) for i in order[:top_k]]

                    fp_logger.info(
                        "Top chromosomes by fp_rate: "
                        + ", ".join([f"chr{c}:{v:.6g}" for c, v in top_pairs])
                    )

                    # vector summary (min/median/max)
                    finite = fp_ch[np.isfinite(fp_ch)]
                    if finite.size:
                        fp_logger.info(
                            f"fp_rate_per_chrom summary: min={np.min(finite):.6g} | median={np.median(finite):.6g} | max={np.max(finite):.6g}"
                        )

                    # optional: print full vector (commented by default)
                    # fp_logger.info("fp_rate_per_chrom (by chrom): " + ", ".join(f"{x:.6g}" if np.isfinite(x) else "nan" for x in fp_ch))

                # decision any_chrom
                res_fp_any=[]
                for sc, scen in enumerate(SCENARIOS):

                    if scen not in strength_map:
                        raise ValueError(f"No strength defined for scenario '{scen}'")

                    strength = float(strength_map[scen])

                    res = estimate_fp_rate_hgki(
                        datos_filtrados_real=datos_filtrados,
                        n_list_per_method=Lsitios[1:],
                        W=W,
                        scenario=scen, 
                        strength=strength,
                        R=R_fp,
                        S=S_fp,
                        alpha=ALFA,
                        decision="any_chrom",
                        ss_scenario= ss_scenarios[sc],
                        hg_test_func=hypergeom_kway_windows_test,
                    )

                    res_fp_any.append(res)

                # Header for the any_chrom block
                det0 = res_fp_any[0].get("details", {})

                fp_logger.info("")
                fp_logger.info("=" * 78)
                fp_logger.info("HGkI FALSE-POSITIVE STUDY (decision=any_chrom)")
                fp_logger.info("=" * 78)
                fp_logger.info(
                    "Design: "
                    f"W={det0.get('W', 'NA')} | alpha={det0.get('alpha', 'NA')} | "
                    f"R={det0.get('R', 'NA')} | S={det0.get('S', 'NA')} | "
                    f"k_methods={det0.get('k_methods', 'NA')} | decision={det0.get('decision', 'NA')}"
                )

                numcroms = len(datos_filtrados[0])
                fwer_ind = 1 - (1 - ALFA)**numcroms

                fp_logger.info(
                    f"Expected FWER under independent chromosome tests "
                    f"(alpha={ALFA}, C={numcroms}): {fwer_ind:.6g}"
                )

                fp_logger.info("-" * 78)

                # --- Summary grouped by scenario ---
                for sc, res in enumerate(res_fp_any):
                    scen = SCENARIOS[sc]
                    det = res.get("details", {})
                    
                    fp_logger.info("")
                    fp_logger.info(f"Scenario: {scen} | strength={det.get('strength', 'NA')}")
                    fp_logger.info("=" * 78)

                    # 0) Global (any_chrom)
                    fp_logger.info(
                        f"Global: fp_rate={res['fp_rate']:.6g} | fp_count={res['fp_count']} | denom={res['denom']}"
                    )

                    # Across universes stats
                    u = np.asarray(res["fp_rate_per_universe"], dtype=float)  # length R
                    mean_u = np.nanmean(u)
                    var_u  = np.nanvar(u, ddof=1) if np.sum(~np.isnan(u)) > 1 else np.nan
                    sd_u   = np.sqrt(var_u) if np.isfinite(var_u) else np.nan
                    n_eff  = int(np.sum(~np.isnan(u)))
                    se_u   = sd_u / np.sqrt(n_eff) if (n_eff > 0 and np.isfinite(sd_u)) else np.nan

                    fp_logger.info(
                        f"Global (FWER, any_chrom): fp_rate={res['fp_rate']:.6g} | fp_count={res['fp_count']} | denom={res['denom']}"
                    )
                    fp_logger.info(
                        f"Across universes: mean={mean_u:.6g} | sd={sd_u:.6g} | var={var_u:.6g} | n_eff={n_eff} | se={se_u:.6g}"
                    )

                    # New: expected number of significant chromosomes per replicate
                    mean_k = res.get("mean_k_sig_per_rep", None)
                    var_k  = res.get("var_k_sig_per_rep", None)
                    if mean_k is not None:
                        fp_logger.info(
                            f"Chroms significant per replicate (k_sig): mean={mean_k:.6g} | var={var_k:.6g}"
                        )

                        kmu = np.asarray(res.get("k_sig_mean_per_universe", []), dtype=float)
                        if kmu.size:
                            fp_logger.info(
                                f"Across universes k_sig mean: mean={np.nanmean(kmu):.6g} | sd={np.nanstd(kmu, ddof=1):.6g} | n_eff={np.sum(~np.isnan(kmu))}"
                            )

           
                logging.info("HGkI false-positive study completed. Program terminated by request.")
                print("HGkI false-positive study completed. See *_FP.log for details.")
                return
                ########## END OF FALSE POSITIVE STUDY FOR HGsI test
            else: ######## v0.3 HGsI test ANALYSIS with REAL DATA
                logging.info(
                    f"Running hypergeometric k-way intersection test (W={W})."
                )
                print(f"Running hypergeometric k-way intersection test (W={W}).\n")
                res = hypergeom_kway_windows_test(
                    datos_filtrados=datos_filtrados,
                    D=W,
                    chrom_index=chrom_index,
                    method_indices=list(range(1, numfiles)),  # k métodos
                )

                for row in res["per_chrom"]:

                    c = row["chrom"] - 1 # chrom is 1-based
                    arrP[c] = row["p_value"]
                    
                    print(row["chrom"], row["N_windows"], row["n_i"], row["k_obs"], row["p_value"])

                    if  args.verbose and row.get("warnings"):
                        print(f"Chromosome {row['chrom']} warnings:")
                        for msg in row["warnings"]:
                            print("  -", msg)
                    
                    for msg in row.get("warnings", []):
                        logging.warning(f"Chromosome {row['chrom']}: {msg}")

        else: ################### T_KL test v0.3 incorporar corrección Davison–Hinkley al cálculo del p-valor p = (B+1)/(R+1) en vez de B/R

            moment1 = datetime.now()
            moment1 = moment1.strftime("%d%m%y_%H%M%S")

            #v0.3 the test for all or for a specific chromosome
            #chrom_index = None if args.chr_id is None else (args.chr_id - 1)

            out_name1=""
            test=  'TKLtest'

            out_name1 = test

            if args.uniform:
                out_name1+='_U'
          
            strkmax=str(args.Kmax)

            if strkmax!='inf':
                out_name1+='_Kmax_' +str(args.Kmax)

            if not args.strict:
                out_name1+='_Perm'
            out_name1+='_'+moment1
            if samefiles:    
                out_name1+='_SAMEFILES'

            
            arrKL=np.full((numcroms),-1.0) # OJO que full si no se define con dtype pilla el tipo por defecto del valor pasado
            
            omedian=np.zeros(numcroms,dtype=int) #Almacena para cada cromosoma el valor medio del estadístico para los datos reales
            emedian=np.zeros(numcroms,dtype=int) #Almacena para cada cromosoma el rango intercuartil de la distribución esperada
            #IQRfactor=3 # Factor de multiplicación del IQR para detectar outliers ligeros (1.5) o fuertes (3)


            if RANDOM: # v0.3 false positives study # SCENARIOS = ["uniform", "center", "left", "right", "extremes" "empirical"]

                logging.info(
                    f"Running FP study on TKL distance test."
                )
                print(f"Running FP study on TKL distance test.\n")

                if args.seed is None:
                    logging.info("Random seed not provided: using system entropy.")
                else:
                    logging.info(f"Random seed set to {args.seed}.")

                if args.chr_id is not None:
                    logging.warning(
                        "FP study ignores --chr_id and is always run on all chromosomes."
                )

                # --- FP logger: separado para el estudio de falsos positivos HGkI ---
                fp_logger = logging.getLogger("KLinterSel.FP")
                fp_logger.setLevel(logging.INFO)
                fp_logfile = f"{out_name1}_FP.log"
                fp_handler = logging.FileHandler(fp_logfile, mode="a", delay=True)
                fp_handler.setFormatter(formatter)

                # Evitar duplicados si se llama más de una vez
                if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == fp_handler.baseFilename
                        for h in fp_logger.handlers):


                    fp_logger.addHandler(fp_handler)
                    
                fp_logger.propagate = False
                
                logging.info(f"TKL FP study enabled. Detailed log: {out_name1}_FP.log")
                fp_logger.info("Starting TKL false-positive study.")
                
                ss_scenarios = ss_global.spawn(len(SCENARIOS))
                strength_map = {
                    "uniform": 1.0,
                    "center": 3.0,
                    "left": 5.0,
                    "right": 5.0,
                    "extremes": 5.0,
                    "empirical": 1.0,
                }


                # decision per_chrom
                res_fp=[]
                for sc, scen in enumerate(SCENARIOS):

                    if scen not in strength_map:
                        raise ValueError(f"No strength defined for scenario '{scen}'")

                    strength = float(strength_map[scen])

                    res = estimate_fp_rate_tkl(
                        datos_filtrados_real=datos_filtrados,
                        n_list_per_method=Lsitios[1:],
                        scenario=scen,
                        strength=strength,
                        R=R_fp,               # tu diseño: expected 1 vez por escenario
                        S=S_fp,             # por defecto 100 réplicas (ajustable)
                        numrs_expected=numrs,
                        numrs_p=numrs//1,      # solo se usa si TEMP=True (si no, es irrelevante)
                        alpha=ALFA,
                        decision="per_chrom",
                        ss_scenario=ss_scenarios[sc],
                        parallel=PARALLEL,
                        num_processes=num_processes,
                        block_size=BLOCK_SIZE,
                        verbose=args.verbose,
                    )


                    res_fp.append(res)

                fp_summary_logger_tkl(
                    fp_logger,
                    "TKL FALSE-POSITIVE STUDY (decision=per_chrom)",
                    res_fp,
                    SCENARIOS,
                    decision="per_chrom",
                    top_k=10
                )


                # decision any_chrom
                res_fp_any=[]
                for sc, scen in enumerate(SCENARIOS):

                    if scen not in strength_map:
                        raise ValueError(f"No strength defined for scenario '{scen}'")

                    strength = float(strength_map[scen])

                    res = estimate_fp_rate_tkl(
                        datos_filtrados_real=datos_filtrados,
                        n_list_per_method=Lsitios[1:],
                        scenario=scen,
                        strength=strength,
                        R=R_fp,
                        S=S_fp,
                        numrs_expected=numrs,
                        numrs_p=numrs//1,
                        alpha=ALFA,
                        decision="any_chrom",
                        ss_scenario=ss_scenarios[sc],
                        parallel=PARALLEL,
                        num_processes=num_processes,
                        block_size=BLOCK_SIZE,
                        verbose=args.verbose,
                    )

                    res_fp_any.append(res)

                fp_summary_logger_tkl(
                    fp_logger,
                    "TKL FALSE-POSITIVE STUDY (decision=any_chrom)",
                    res_fp_any,
                    SCENARIOS,
                    decision="any_chrom",
                    top_k=10,
                    alpha=ALFA,
                    numcroms=numcroms
                )

                logging.info("TKL false-positive study completed. Program terminated by request.")
                print("TKL false-positive study completed. See *_FP.log for details.")
                return

                ########## END RANDOM for FALSE POSITIVE STUDY FOR TKL test
            else: ######## v0.3 TKL test ANALYSIS with REAL DATA

                #### CALCULAMOS PARA CADA CROMOSOMA LA DISTRIBUCIÓN DE DISTANCIAS ENTRE LOS SITIOS CANDIDATOS ###

                print(f'\nCHR\tTOTS\t',end='')
                for i in range(1,numfiles):
                    print(f'SEL-{i}\t',end='')
                print(f'KL\tPr\toQ2\teQ2\n')
                
                for c in range(numcroms):
                    
                    # v0.2 Nov 2025 Ordena antes (aunque posiblemente ya está ordenado) y calcula el rango para cada cromosoma
                    datos_filtrados[0][c].sort() # el coste de ordenación es mínimo O(nlogn) no llega a 1 segundo en total en listas de 50000 snps y 20-30 cromosomas
                    min_val = int(datos_filtrados[0][c][0])
                    max_val=int(datos_filtrados[0][c][-1])

                    rango=max_val-min_val + 1# Máxima distancia entre marcadores
                    data_per_crom=[]
                    
                    nonEmpty=0
                    maxsites = 0

                    for i in range(1,numfiles):

                        if Lsitios[i][c]:
                            nonEmpty+=1

                        data_per_crom.append(datos_filtrados[i][c]) #Para cada cromosoma c: Lista de arrays, cada array de la lista corresponde a un fichero

                        #print(Lsitios[i][c])
                        
                    if nonEmpty<2:
                        nonEmpty=0

                    if nonEmpty:
                        maxsites = numdists(*data_per_crom)

                        #Estimamos el número de GiB de memoria
                        TEMP=False # Este modo solo se usa si no hay espacio para almacenar (numrs, maxsites)

                        # CALCULA RAM DISPONIBLE
                        forma_array = (numrs, maxsites)
                        mem, err, reqram = procesar_bloque(forma_array)
                        if not mem: #Si no hay memoria suficiente para almacenar todas las réplicas pasamos al modo TEMP que no las almacena

                            try: #Comprobamos que sí hay memoria para el número total de distancias si no lanzamos excepción y terminamos
                                #mensaje_error_1 = f"Consider using the --Kmax or --notest arguments.\n"
                                #numrs = max(MinPerm,numrs//10)
                                forma_array = (maxsites)
                                #print('Memory problem. The number of permutations is reduced by an order of magnitude. '+mensaje_error_1)
                                mem2, _ ,_= procesar_bloque(forma_array)
                                if not mem2:
                                    errmensaje =err
                                    
                                    raise MemoryError(errmensaje)
                                else:
                                    TEMP=True

                            except MemoryError as e:
                                #mensaje_error_2 = f"Error: {e}\nPlease refer to the 'memory.log' file for more details."
                                #logging.error(mensaje_error_completo)  # Mensaje personalizado de error en el log
                                #print(mensaje_error_completo)
                                logging.error(mensaje)
                                print(f"\nError: {e}")
                                print(f"Please refer to the {Path(handler.baseFilename).name} file for more details.")
                                sys.exit(1)

                        avertot_distances=np.zeros(maxsites) # Inicializar
                        if not TEMP: # hay suficiente memoria para manejar todas las réplicas
                            if args.uniform: #v0.2 Nov 2025
                                # NULA UNIFORME
                                rtot_distances = generate_uniform_rtot_distances(numrs, numfiles, Lsitios, c, maxsites, min_val, max_val)
                            else:
                            # 25 july 25 vectorized improvement
                                rtot_distances =generate_rtot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites)

                            avertot_distances = np.mean(np.vstack(rtot_distances), axis=0)
                            
                        else: #NO SE ALMACENAN RÉPLICAS

                            if PARALLEL and not args.uniform:
                                print(f'Moving to memory management functions to handle {round(reqram)} GiB')
                                print(f'CALCULATING EXPECTED DISTRIBUTION')
                                avertot_distances = calcular_avertot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites,maxproc=num_processes)
                            else:

                                for r in range(numrs):
                                    if args.uniform: #v0.2 Nov 2025
                                        rmuestra = [
                                            np.random.randint(min_val, max_val + 1, size=Lsitios[i][c])
                                            for i in range(1, numfiles)
                                        ]
                                    else:
                                        rmuestra = [
                                            np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)
                                            for i in range(1, numfiles)
                                        ]

                                    avertot_distances +=totdist(*rmuestra, end=maxsites) # totdist ya devuelve ordenado el array por eso avertot_distances ya está ordenado
                                avertot_distances/=numrs
                        
                        #DISTRIBUCIÓN OBSERVADA

                        tot_distances[c] = totdist(*data_per_crom, end=maxsites) # totdist ya devuelve ordenado el array
                            
                        

                        #Calcula la entropía relativa entre la distribución real y la esperada por azar
                        arrKL[c] = RelEntr(tot_distances[c],avertot_distances) # Cantidad de información perdida cuando avertot_distances se usa para aproximar tot_distances[c]

                        # Calcula la mediana de la distribución de distancias observadas y la de la esperada
                        
                        omedian[c]=np.percentile(tot_distances[c],50)
                        emedian[c]=np.percentile(avertot_distances,50)
                        #Q1 = np.percentile(avertot_distances,25)
                        #IQR[c]=np.percentile(avertot_distances,75) - Q1
                        #limite_inferior = Q1-IQRfactor*IQR[c] # Valor menor 3 veces por debajo del rango intercuartil

                        if omedian[c]<= emedian[c]:

                            '''
                            - REMUESTREO:

                            '''
                                
                        # Generar las muestras sin reemplazo dentro de cada muestra
                        #np.random.seed(42)
                                           
                            if args.uniform: # muestrea posiciones de una uniforme

                                if not TEMP:
                                    arrP[c] = calculate_vectorized(rtot_distances, avertot_distances, arrKL, emedian, c)
                                else:

                                    arrP[c] = 0.0 # Antes de acumular, no necesario porque se inicializa a zeros pero por seguridad reduntante
                                    

                                    for _ in range(numrs):
                                        Umuestra = []
                                        for i in range(1, numfiles):
                                            Umuestra.append(
                                                np.random.randint(
                                                    low=min_val,
                                                    high=max_val + 1,
                                                    size=Lsitios[i][c]
                                                )
                                            )

                                        rtot_distances = totdist(*Umuestra, end=maxsites)

                                        if RelEntr(rtot_distances,avertot_distances) >= arrKL[c] and np.percentile(rtot_distances,50) <= emedian[c]:
                                            
                                            arrP[c] += 1

                                    #arrP[c] /= numrs
                                    arrP[c] = (arrP[c]+1) / (numrs + 1.0) #v0.3 corrección Davison–Hinkley

                                  
                            elif REMUESTREO: # muestrea directamente posiciones reales

                                '''
                                La divergencia de la distribución de distancias entre candidatos con la distribución esperada (media) por azar
                                tiene valor arrKL[c] ¿Cuan probable es obtener un valor igual o mayor que este entre una divergencia obtenida por azar
                                y la esperada por azar?
                                '''

                                if not TEMP: # Las permutaciones están almacenadas en rtot_distances

                                    arrP[c] = calculate_vectorized(rtot_distances, avertot_distances, arrKL, emedian, c)
                                    
                                else: #Las permutaciones no están almacenadas y se vuelven a hacer para calcular el valor p

                                    if PARALLEL:
                                        print(f'CALCULATING p-VALUES')
                                        arrP[c] = calcular_arrP_paralelo_bloques(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=num_processes,block_size=BLOCK_SIZE)
                                    else:
                                    
                                        for r in range(numrs):

                                            rmuestra = [
                                                np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)
                                                for i in range(1, numfiles)
                                            ]

                                            rtot_distances = totdist(*rmuestra, end=maxsites)

                                            if RelEntr(rtot_distances,avertot_distances) >= arrKL[c] and np.percentile(rtot_distances,50) <= emedian[c]: 
                                            
                                                arrP[c] +=1

                                        #arrP[c]/=numrs
                                        arrP[c] = (arrP[c] +1.0) / (numrs + 1.0) #v0.3 corrección Davison–Hinkley

                            else:

                                arrP[c]=1

                        else: # Si la mediana observada no es menor que la esperada
                            arrP[c]=1
                    else: # No hay candidatos para este cromosoma
                        
                        arrP[c]=1

                    print(f'{c+1}\t{Lsitios[0][c]}\t',end='')
                    for i in range(1,numfiles):
                        print(f'{Lsitios[i][c]}\t',end='')
                    print(f'{np.format_float_positional(arrKL[c], precision=4)}\t{format_pvalue(arrP[c], nperm=numrs)}\t{omedian[c]}\t{emedian[c]}\n')

                    if FIGURE and (args.paint=='all' or args.paint==str(c+1)): # tot_distances[c],avertot_distances
                        fixed_max_xvalue = None
                        
                        if args.max_xvalue is not None: 
                            fixed_max_xvalue = args.max_xvalue * 1e6  # Convertir de Mb a bases

                        # Valor fijo para el eje Y (proporcionado directamente como float, sin conversión)
                        fixed_max_yvalue = args.max_yvalue

                        nbins, barwidth = calculate_bins(avertot_distances)
                        # Crear un DataFrame combinado
                        df = pd.DataFrame({
                            'value': np.concatenate([tot_distances[c], avertot_distances]),
                            'array': ['OBSERVED'] * len(tot_distances[c]) + ['EXPECTED'] * len(avertot_distances)
                        })

                        # Calcular el máximo valor entre los dos arrays
                        max_value = max(np.max(tot_distances[c]), np.max(avertot_distances))
                        # Determinar la unidad (kb o Mb). Opcional: descomentar para forzar una unidad específica
                        # unit_name, unit_divisor = determine_unit(max_value, manual_unit='kb')  # Forzar kb
                        # unit_name, unit_divisor = determine_unit(max_value, manual_unit='mb')  # Forzar Mb
                        #unit_name, unit_divisor = determine_unit(max_value)  # Selección automática
                        unit_name, unit_divisor = determine_unit(max_value if fixed_max_xvalue is None else fixed_max_xvalue)  # Selección automática

                        # Obtener la unidad adecuada para la escala en las unidades seleccionadas
                        unit_in_units = get_nice_scale(max_value, unit_divisor, fixed_max_value=fixed_max_xvalue) / unit_divisor  # unidad en kb o Mb
                        unit = unit_in_units * unit_divisor  # unidad en la escala original (para get_ticks)
                        #ticks = get_ticks(max_value, unit)
                        ticks = get_ticks(max_value if fixed_max_xvalue is None else fixed_max_xvalue, unit, fixed_max_value=fixed_max_xvalue)

                        # Crear el gráfico de distribución con frecuencias
                        p = sns.displot(data=df, x='value', col='array', kind='hist',
                                        height=5, aspect=2, stat='probability',
                                        bins=nbins)  # Ajustar el número de bins según sea necesario

                        # Calcular los límites del eje Y para que sean iguales en ambos subplots
                        ylims = [ax.get_ylim() for ax in p.axes.flat]
                        max_ylim = max([ylim[1] for ylim in ylims])

                         # Usar fixed_max_yvalue si está establecido, de lo contrario usar max_ylim
                        y_max = fixed_max_yvalue if fixed_max_yvalue is not None else max_ylim
                        
                        # Gestionar formato ejes
                        for ax in p.axes.flat:
                            for patch in ax.patches:
                                patch.set_facecolor('none')
                                patch.set_edgecolor('black')
                                patch.set_linewidth(1)  # Grosor del borde

                            #Configurar eje x

                            #ax.set_xlim(-barwidth, max_value)
                            x_max = fixed_max_xvalue if fixed_max_xvalue is not None else max_value
                            ax.set_xlim(-barwidth, x_max)
                            ax.spines['left'].set_position(('data', -barwidth))

                            # Configurar eje Y con el valor máximo fijo si se proporciona
                            ax.set_ylim(0, y_max)

                            # Configurar los ticks del eje X en intervalos adecuados
                            
                            ax.set_xticks(ticks)
                            # Establecer las etiquetas de los ticks en unidades adecuadas (Mb o kb)
                            labels = format_tick_labels(ticks, unit_divisor)
                            ax.set_xticklabels(labels)

                            # Opcional: mostrar la unidad en el eje X
                            ax.set_xlabel(unit_name, fontsize=11)

                            # Configurar ticks del eje Y si se fija el valor máximo
                            if fixed_max_yvalue is not None:
                                # Establecer ticks cada 0.05 si el máximo es 0.2 (ajusta según tus necesidades)
                                y_ticks = np.arange(0, fixed_max_yvalue + 0.01, fixed_max_yvalue / 5)
                                y_ticks = np.unique(y_ticks)  # Eliminar duplicados si fixed_max_yvalue / 5 no divide uniformemente
                                ax.set_yticks(y_ticks)
                            
                            # Configurar cada subplot para mostrar solo las líneas de los ejes X e Y
                            # Eliminar los spines derecho y superior
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)

                            # Eliminar los títulos de los ejes
                            #ax.set_xlabel('')
                            ax.set_ylabel('')
                        #Personalizar títulos de cada gráfico
                        #titles = ['OBSERVED', 'EXPECTED']
                        titles = ['', '']
                        for ax, title in zip(p.axes.flat, titles):
                            ax.set_title(title, fontsize=11, fontweight='bold', pad=7)

                        #plt.tight_layout()
                        plt.show()
                
    else: # if not TEST
        arrP=np.ones(numcroms)
        ALFA=1.0

    ######################## INTERSECCIONES #######################################
        
    ### Calculamos la INTERSECCIÓN entre métodos solo para los cromosomas significativos o
    # si queremos para todos los cromosomas basta poner ALFA=1 o simplemente --notest

    keyname='SEL_'
    if not RANDOM and not FIGURE and intersec: #Tanto la opción --rand como la opción --paint inactivan el cálculo de intersecciones
        
        print('\n### INTERSECTION ANALYSIS ###')
        if not args.chr_id: # No se indicó cromosoma específico
            for c in range(numcroms):

                data_per_crom=[]
                
                nonEmpty=0
                
                for i in range(1,numfiles):

                    if Lsitios[i][c]:
                        nonEmpty+=1
                    data_per_crom.append(datos_filtrados[i][c])

                if nonEmpty<2:
                    nonEmpty=0
                #print(arrP[c], ALFA, nonEmpty)
                if arrP[c]<=ALFA and nonEmpty:
                    
                    #cd[c], kset[c] =intersec_Dn_Opt_no_sort(*data_per_crom, D=distance,keyname=keyname)
                    cd[c], kset[c] =intersec_Dn_Opt_sorted(*data_per_crom, D=distance,keyname=keyname) #v0.2 acr Nov 2025
                    #print(f"Chrom {c+1} kset[c] =", kset[c])
                    #if c==17:
                        #sys.exit(6)

        else:

            c= chrom_index # = args.chr_id - 1
            data_per_crom=[]
            
            nonEmpty=0
            
            for i in range(1,numfiles):

                if Lsitios[i][c]:
                    nonEmpty+=1
                data_per_crom.append(datos_filtrados[i][c])

            if nonEmpty<2:
                nonEmpty=0
            #print(arrP[c], ALFA, nonEmpty)
            if arrP[c]<=ALFA and nonEmpty:

                #cd[c], kset[c] =intersec_Dn_Opt_no_sort(*data_per_crom, D=distance,keyname=keyname)
                cd[c], kset[c] =intersec_Dn_Opt_sorted(*data_per_crom, D=distance,keyname=keyname) #v0.2 acr Nov 2025

                #print(f"Chrom {c+1} kset[c] =", kset[c])
                #sys.exit(7)

            
    #####################################################################################
    ### ########################        OUTPUT                  #########################
    #####################################################################################
    
    dirsalida="KLinterSel_Results"
    dirsalida = os.path.join(dirdatos,dirsalida)
    mkdir(dirsalida)

    #Para posterior construcción del nombre de los ficheros de resultados
    formatted_distance = f"{distance:.0E}".replace('+', '').replace('E0', 'E') # to print x with 3 decimals: f'x:.3E'
    
    #Moment
    moment = datetime.now()
    moment = moment.strftime("%d%m%y_%H%M%S")

    if not TEST:
        test=  'NoTest'
        out_name3 = test
        strkmax=str(args.Kmax)
        if strkmax!='inf':
            out_name3+='_Kmax_' +str(args.Kmax)

        if not args.strict:
            out_name3+='_Perm'
        out_name3+='_'+moment
        if samefiles:    
            out_name3+='_SAMEFILES'
    
    elif args.HG:
        out_name3 = out_name2
    else:
        out_name3 = out_name1
    

    if TEST:

        #v0.3

        # Spatial uniformity ranking metrics (SURM)
        metrics = uniformity_metrics_per_chrom_scipy(datos_filtrados[0])
        uniformity = rank_chromosomes(metrics, key="composite")

        if args.HG: # Write the results for the HGsI test
            any_missing=False
            delimiter='\t'
            ruta=os.path.join(dirsalida,out_name2+'.tsv') # tab-separated values
            with open(ruta, 'w') as f:
                f.write(f'{nombre[0]}{delimiter}{np.sum(Lsitios[0])}{delimiter}SNPs\n')
                for i in range(1,numfiles):
                    nsites=np.sum(Lsitios[i])

                    f.write(f'{nombre[i]} ({keyname}{i}){delimiter}{nsites}{delimiter}SNPs\n')

                f.write(f'Number of permutations{delimiter}{numrs}\n')
                f.write(f'CHR{delimiter}TOTALS{delimiter}Nc{delimiter}')
                for i in range(1,numfiles):
                    f.write(f'SNPs{i}{delimiter}n{i}{delimiter}')

                f.write(f'kobs{delimiter}E_k_null{delimiter}Pr{delimiter}URank{delimiter}Uclasf{delimiter}notU\n')

                for row in res["per_chrom"]:

                    c = row["chrom"] - 1   # chrom es 1-based
                    u = uniformity[c]
                    rank = u["composite_score"] # v0.3 rank: 0 perfect uniform, 1 strong deviation
                    cls  = u["uniformity_class"] # clasificación close, media o strong deviation
                    #strong = strong_metrics(u["metric_classes"],order=["ks_stat", "cvm_stat", "cv_counts"]) #clasificación por métrica
                    NotUmetric = get_metrics(u["metric_classes"],class_type=CLOSE,exclude=True,order=METRIC_ORDER) # clasificación por metrica

                    if row["missing"]: # 
                        any_missing = True

                    N = int(row["N_windows"])
                    nvec = row["n_i"]  # lista de k tamaños (número de ventanas candidatas por método)
                    # Aproximación E(K) bajo nulo (k-way intersection)
                    if N > 0 and all(int(x) >= 0 for x in nvec):
                        k = len(nvec)
                        # si algún n_i es 0, el producto será 0 -> E(K)=0
                        prod_n = 1.0
                        for x in nvec:
                            prod_n *= float(x)
                        E_k_null = prod_n / (float(N) ** (k - 1)) if k >= 2 else float(nvec[0])
                    else:
                        E_k_null = np.nan
                    
                    f.write(f'{row["chrom"]}{delimiter}{Lsitios[0][c]}{delimiter}{row["N_windows"]}{delimiter}')
                    for i in range(1,numfiles):
                        f.write(f'{Lsitios[i][c]}{delimiter}{row["n_i"][i-1]}{delimiter}')


                    f.write(f'{row["k_obs"]}{delimiter}'
                            f'{E_k_null:.6g}{delimiter}'
                            f'{fmt_p(row["p_value"])}{delimiter}{rank:.2g}{delimiter}{cls}'
                    )
                    # v0.3 La información sobre qué medidas indicaron desviación fuerte
##                    if (u["uniformity_class"] == CLOSE and strong) or row["p_value"]<=ALFA:
##                        f.write(f"{delimiter}{delimiter.join(strong)}")

                    f.write(f"{delimiter}{delimiter.join(NotUmetric)}")                  
                    f.write(f'\n')

                if any_missing:
                    f.write(
                        "\nWARNING: Some candidate SNPs were not present in the original SNP "
                        "universe and were ignored. Please check the log file for details."
                    )

        else:
            # Write the results for the KL test
            delimiter='\t'
            ruta=os.path.join(dirsalida,out_name1+'.tsv') # tab-separated values

            with open(ruta, 'w') as f:
                
                f.write(f'{nombre[0]}{delimiter}{np.sum(Lsitios[0])}{delimiter}SNPs\n')

                for i in range(1,numfiles):
                    nsites=np.sum(Lsitios[i])
                    if nsites==maxsnps_perfile[i]:
                        f.write(f'{nombre[i]} ({keyname}{i}){delimiter}{nsites}{delimiter}SNPs\n')
                    else:
                        f.write(f'{nombre[i]} ({keyname}{i}){delimiter}{nsites}{delimiter}SNPs{delimiter}after grouping from {maxsnps_perfile[i]}\n')
                    
                f.write(f'Number of permutations{delimiter}{numrs}\n')
                f.write(f'CHR{delimiter}TOTALS{delimiter}')
                for i in range(1,numfiles):
                    f.write(f'SEL-{i}{delimiter}')

                f.write(f'KL{delimiter}Pr{delimiter}oQ2{delimiter}eQ2{delimiter}URank{delimiter}Uclasf{delimiter}notU\n')
                for c in range(numcroms):
                    #v0.3 SURM
                    u = uniformity[c]
                    rank = u["composite_score"] # v0.3 rank: 0 perfect uniform, 1 strong deviation
                    cls  = u["uniformity_class"] # clasificación close, media o strong deviation
                    NotUmetric = get_metrics(u["metric_classes"],class_type=CLOSE,exclude=True,order=METRIC_ORDER) # clasificación por metrica
                    f.write(f'{c+1}{delimiter}{Lsitios[0][c]}{delimiter}')
                    for i in range(1,numfiles):
                        f.write(f'{Lsitios[i][c]}{delimiter}')
                    f.write(f'{np.format_float_positional(arrKL[c], precision=4)}{delimiter}{format_pvalue(arrP[c], nperm=numrs)}{delimiter}{omedian[c]}{delimiter}{emedian[c]}{delimiter}{rank:.2g}{delimiter}{cls}')
                    # v0.3 La información sobre qué medidas indicaron desviación fuerte
##                    if (u["uniformity_class"] == CLOSE and strong) or arrP[c]<=ALFA:
##                        f.write(f"{delimiter}{delimiter.join(strong)}")
                    f.write(f"{delimiter}{delimiter.join(NotUmetric)}")
                        
                    f.write(f'\n')

    # Name for the interseccion file
    if not RANDOM and intersec:
        text= 'INTERSEC_'+test+ '_D'+formatted_distance
        out_name3= out_name3.replace(test,text)+'.tsv'
        ruta2=os.path.join(dirsalida,out_name3)
        delimiter2='\t'

        # ESCRIBIR LAS INTERSECCIONES
        # Obtener las claves del primer diccionario no vacío
        keys=[]
        for scd in cd:
            if scd:
                keys = list(scd.keys())
                break
        if keys:
            # encabezados
            K=numfiles-1
            header = f'CHR{delimiter2}Distance{delimiter2}{delimiter2.join(keys)}{delimiter2}{K}-intersec-sites'
            
            with open(ruta2, 'w') as f:
                f.write(f'{nombre[0]}{delimiter2}{np.sum(Lsitios[0])}{delimiter2}SNPs\n')
                for i in range(1,numfiles):
                    nsites=np.sum(Lsitios[i])
                    if nsites==maxsnps_perfile[i]:
                        f.write(f'{nombre[i]} ({keyname}{i}){delimiter2}{nsites}{delimiter2}SNPs\n')
                    else:
                        f.write(f'{nombre[i]} ({keyname}{i}){delimiter2}{nsites}{delimiter2}SNPs{delimiter2}after grouping from {maxsnps_perfile[i]}\n')     
                
                f.write(f'{header}\n')

                for c in range(numcroms):
                    if cd[c]: #Crompueba que este cromosoma tiene diccionario
                        # Determinar el número máximo de filas
                        max_rows = max(len(cd[c][key]) for key in keys)
                        #Actualiza con el set de sitios julio 2025
                        nksites=len(kset[c])
                        max_rows=max(max_rows, nksites)
                        kset[c]=sorted(kset[c]) # ordena los sitos kset de cada cromosoma

                        #print(f"Chrom {c+1} kset[c] =", kset[c])

                        # Imprimir los datos

                        for row in range(max_rows):
                            row_data = []
                            for key in keys:
                                if row < len(cd[c][key]): # en cada número de fila solo añade ese número de elemento de cada key cd[c][key][row]
                                    row_data.append(str(cd[c][key][row]))
                                else:
                                    row_data.append('')
                            f.write(f"{c+1}{delimiter2}{formatted_distance}{delimiter2}")
                            f.write(delimiter2.join(row_data))
                            if row<nksites:
                                f.write(f"{delimiter2}{kset[c][row]}\n")
                            else:
                                f.write("\n")
                            
                        f.write('\n')

    print("\nKLinterSel DONE\n") 


if __name__ == "__main__":
    main()
