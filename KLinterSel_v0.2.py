'''
Author: ACR
Program Name: KLinterSel
Last Update 28 Nov 2025: v0.2

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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#from scipy.special import rel_entr # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html
from scipy.stats import chi2

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

def calcular_arrP_paralelo(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=MAXPROC):
    '''
    Usage: arrP[c] = calcular_arrP_paralelo(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian)
    '''
    # Crear una lista de argumentos para cada réplica
    args_list = [
        (r, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian)
        for r in range(numrs)
    ]

    # Usar multiprocessing para paralelizar
    with Pool(processes=maxproc) as pool:
        resultados = pool.map(procesar_replica_p, args_list)

    # Calcular arrP[c]
    arrP = sum(resultados) / numrs
    return arrP

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
    arrP = count_satisfy_condition / numrs
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

##def RelEntr_sin_controles(arr1,arr2):
##
##    assert len(arr1) == len(arr2), "Arrays must have the same length"
##    
##    # Normalizar los arrays
##    P = arr1 / arr1.sum()
##    Q = arr2 / arr2.sum()
##
##    # Calcular la entropia relativa KL
##    rel_entr_values = rel_entr(P, Q) # Cantidad de información que se pierde cuando Q se usa para aproximar P
##    
##    kl_div = np.sum(np.where(Q == 0, 0, rel_entr_values)) # corrige valores inf causados por Q==0
##
##    return kl_div

# v0.2 Nov 2025
##def RelEntr(arr1, arr2):
##    """
##    KL-like discrepancy between two non-negative vectors arr1 and arr2.
##    Devuelve 0.0 en casos degenerados (una sola distancia, suma = 0, etc.).
##    """
##
##    # Ambos deben tener igual longitud
##    assert len(arr1) == len(arr2), "Arrays must have the same length"
##
##    # Caso degenerado: arrays demasiado cortos
##    if len(arr1) < 2:
##        # No tiene sentido calcular KL-like con 1 sola distancia
##        return 0.0 # 0.0 si se prefiere anular el efecto o np.nan si preferimos indentificar el caso
##
##    sum1 = arr1.sum()
##    sum2 = arr2.sum()
##
##    # Caso en que arr1 o arr2 suman 0 (p.ej. arr = [0,0,...,0])
##    if sum1 <= 0 or sum2 <= 0:
##        # No se puede normalizar: perfil degenerado
##        return 0.0   # o np.nan si preferimos indentificar el caso
##
##    # Normalización
##    P = arr1 / sum1
##    Q = arr2 / sum2
##
##    # Evitar divisiones por cero o logs de cero:
##    # Donde P=0, el término KL es 0 por convención
##    # Donde Q=0 pero P>0, KL tiende a +inf  lo forzamos a un valor grande o lo tratamos como evidencia
##    with np.errstate(divide='ignore', invalid='ignore'):
##        rel_entr_values = P * np.log(P / Q)
##
##    # Reemplazar NaN o inf:
##    # NaN (0*log(...)) se vuelve 0
##    # inf (P>0, Q=0) podría interpretarse como discrepancia máxima.
##    # Pero lo más seguro en nuestro contexto es tratarlo como 0 si Q[i]==0 y P[i]==0,
##    # y también si P[i]>0 y Q[i]==0.
##    rel_entr_values = np.where(np.isnan(rel_entr_values), 0.0, rel_entr_values)
##
##    # Donde Q==0 pero P>0 sería discrepancia extrema pero
##    # lo anulamos si quisiéramos identificarlo como discrepancia extrema usaríamos np.inf en vez de 0.
##    rel_entr_values = np.where((Q == 0) & (P > 0), 0.0, rel_entr_values) # corrige valores inf causados por Q==0
##
##    # Opción si se hubiera mantenido los valores de np.inf que no es el caso:
##    #if np.isinf(rel_entr_values).any():
##        # Opción A: Devuelve un valor enorme (muy penalizado)
##        # return 1e12
##        
##        # Opción B: Considera que la comparación es totalmente divergente
##        #return np.nan
##
##    return float(rel_entr_values.sum())

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
    rtot_array = np.array(rtot_distances)

    # Calcular percentiles y condiciones
    conditions_met = np.array([
        RelEntr(rtot_distances[r], avertot_distances) >= arrKL[c] and
        np.percentile(rtot_distances[r], 50) <= emedian[c]
        for r in range(numrs)
    ])

    # Contar cuántas condiciones se cumplen
    arrP_increment = np.sum(conditions_met)
    
    arrP_increment / numrs

    return arrP_increment / numrs

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


##def RelEntr_vectorized(arr1_3d, arr2):
##    """
##    Calcula la divergencia KL entre cada array en una matriz 3D y un array de referencia.
##
##    Parámetros:
##    - arr1_3d: Matriz 3D de forma (numrs, longitud_array)
##    - arr2: Array 1D que se usará como referencia para cada array en arr1_3d.
##
##    Retorna:
##    - kl_divs: Array de divergencias KL para cada conjunto de datos en arr1_3d.
##
##    Usage: kl_divs_result = RelEntr_vectorized(rtot_array_3d, avertot_distances)
##    """
##    # Convertir la lista de ndarrays a un solo ndarray 3D
##    rtot_array = np.array(arr1_3d)
##    # Verificar que las dimensiones coinciden
##    assert rtot_array.shape[1] == arr2.shape[0], "Los arrays deben tener longitudes compatibles"
##
##    # Normalizar arr1 y arr2 para cada conjunto de datos a lo largo de la primera dimensión
##    P = rtot_array / rtot_array.sum(axis=1, keepdims=True)
##    Q = arr2 / arr2.sum()
##
##    # Asegúrate de que Q es un array 2D para las operaciones broadcast
##    Q_expanded = np.expand_dims(Q, axis=0)  # Cambiar forma para broadcast (1, longitud_array)
##
##    # Calcular la divergencia KL para cada par de arrays
##    rel_entr_values = rel_entr(P, Q_expanded)
##    kl_divs = np.sum(np.where(Q_expanded == 0, 0, rel_entr_values), axis=1)
##
##    return kl_divs

##def RelEntr_vectorized_blocked(arr1_3d, arr2, block_size=100):
##    # Convertir la lista de ndarrays a un solo ndarray 3D
##    rtot_array = np.array(arr1_3d)
##    num_samples = rtot_array.shape[0]
##    kl_divs = np.zeros(num_samples)
##
##    for i in range(0, num_samples, block_size):
##        end_idx = min(i + block_size, num_samples)
##        block = rtot_array[i:end_idx]
##        P = block / block.sum(axis=1, keepdims=True)
##        Q = arr2 / arr2.sum()
##        Q_expanded = np.expand_dims(Q, axis=0)
##        rel_entr_values = rel_entr(P, Q_expanded)
##        kl_divs[i:end_idx] = np.sum(np.where(Q_expanded == 0, 0, rel_entr_values), axis=1)
##
##    return kl_divs

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

    return arrP_increment / numrs


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
    dec = len(str(nperm)) - 1
    return f"{p:.{dec}f}"


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
                        help='compute intersection without performing KL test')


    # Argumento con nombre para no aplicar la regla strict: si paso el argumento --permissive pone la variable de dest que es strict a False

    parser.add_argument('--permissive', action='store_false', dest='strict',
                    help='avoid strict rule for redundancy')

    # Intersecciones solo para el cromosoma indicado
    parser.add_argument('--chr-id',
                        type=positive_int,  # Usar la función de validación
                        default=None,      # Valor por defecto es None
                        help='Cromosome number (integer >= 1)')

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
    # Parsear los argumentos
    args = parser.parse_args()

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
    if random_files: #En vez de usar los datos obtenidos genera datos aleatorios basados en la estructura de los datos
        for i in range(1,numfiles):
            datos_filtrados[i] = genrandomdata(datos_filtrados[0],datos_filtrados[i])


    # v0.2 Opción conservadora que elimina un set de candidatos si coincide con el de otro método (implicaría métodos muy similares o iguales)
    else: # En el caso de no ser random_files


        if args.strict:

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

    # Listas para cada fichero (posición 0 es el fichero total, las otras las de los métodos)
    #cada elemento de la lista contiene una lista con el número de sitios totales (primera lista) o significativos por cromosoma
    Lsitios=[]
    for data in datos_filtrados: # Cada elemento de datos filtrados corresponde a los datos de un fichero
        Lsitios.append([fila.size for fila in data]) #Lsitos tiene tantas filas como ficheros y tantas columnas como cromosomas. Los valores de celdas (dim 3) son el número de snps en ese cromosoma (columna dim 2) y fichero (fila dim 1)

    numcroms=len(Lsitios[0])

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
    print(f'The total number of SNPs in the original data {nombre[0]} is {maxsnps_perfile[0]}')
    for i in range(1,numfiles):
        assert numcroms==len(Lsitios[i])
        print(f'The number of significant SNPs in {nombre[i]} before filtration was {maxsnps_perfile[i]} and after was {np.sum(Lsitios[i])}. If there were more than {args.Kmax} significant SNPs, those within clusters closer than {distance} nucleotides were filtered out.')

    ########
    ######## CÁLCULO DEL ESTADÍSTICO Y REMUESTREO 
    ########

    # CALCULO NÚMERO DE CORES-1 POR SI HAY PARALELIZACIÓN

    num_processes=max(1,psutil.cpu_count(logical=False)-1)

    if TEST:

        print('\n### START OF CHROMOSOME ANALYSIS ###')
        
        arrKL=np.full((numcroms),-1.0) # OJO que full si no se define con dtype pilla el tipo por defecto del valor pasado

        arrP=np.zeros(numcroms)
        omedian=np.zeros(numcroms,dtype=int) #Almacena para cada cromosoma el valor medio del estadístico para los datos reales
        emedian=np.zeros(numcroms,dtype=int) #Almacena para cada cromosoma el rango intercuartil de la distribución esperada
        #IQRfactor=3 # Factor de multiplicación del IQR para detectar outliers ligeros (1.5) o fuertes (3)

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
                
                if not RANDOM: #DISTRIBUCIÓN OBSERVADA

                    tot_distances[c] = totdist(*data_per_crom, end=maxsites) # totdist ya devuelve ordenado el array
                    
                else: #Divergencia entre una distribución al azar y la media
                    #Genera distancias al azar
                    muestras=[]
                    for i in range(1,numfiles):
                        muestras.append(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False))

                    tot_distances[c] = totdist(*muestras, end=maxsites)

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

                            arrP[c] = 0.0 # no necesario porque se inicializa a zeros pero por seguridad reduntante

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

                            arrP[c] /= numrs

                          
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

                                arrP[c]/=numrs

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
    if not RANDOM and not FIGURE: #Tanto la opción --rand como la opción --paint inactivan el cálculo de intersecciones
        
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

            c= args.chr_id - 1
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
            
    #Crea el directorio de salida si no existe
    dirsalida="KLinterSel_Results"
    dirsalida = os.path.join(dirdatos,dirsalida)
    mkdir(dirsalida)

    #Construye el nombre del fichero de resultados
    formatted_distance = f"{distance:.0E}".replace('+', '').replace('E0', 'E') # to print x with 3 decimals: f'x:.3E'
    #Moment
    moment = datetime.now()
    moment = moment.strftime("%d%m%y_%H%M%S")
    out_name='testTKL'
    strkmax=str(args.Kmax)
    if strkmax!='inf':
        out_name+='_Kmax_' +str(args.Kmax)
    
    if not args.strict:
        out_name+='_Perm'

    out_name+='_'+moment
    
    samefiles=True

    for i in range(1,numfiles-1):

        if nombre[i].upper()!=nombre[i+1].upper():
            samefiles=False
            break

    if samefiles:    
        out_name+='_SAMEFILES'

    out_name2 = out_name
    
    if args.uniform:
        out_name+='_UNIFORM'

    if RANDOM:
        out_name+='_RANDOM'
    if TEST:
       # Write the results for the KL test
        delimiter='\t'
        ruta=os.path.join(dirsalida,out_name+'.tsv') # tab-separated values

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

            f.write(f'KL{delimiter}Pr{delimiter}oQ2{delimiter}eQ2\n')
            for c in range(numcroms):
                f.write(f'{c+1}{delimiter}{Lsitios[0][c]}{delimiter}')
                for i in range(1,numfiles):
                    f.write(f'{Lsitios[i][c]}{delimiter}')
                f.write(f'{np.format_float_positional(arrKL[c], precision=4)}{delimiter}{format_pvalue(arrP[c], nperm=numrs)}{delimiter}{omedian[c]}{delimiter}{emedian[c]}\n')

    # Name for the interseccion file
    if not RANDOM:
        text= 'INTERSEC'+ '_D'+formatted_distance
        out_name2= out_name2.replace('testTKL',text)+'.tsv'
        ruta2=os.path.join(dirsalida,out_name2)
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

    print("KLinterSel DONE") 


if __name__ == "__main__":
    main()
