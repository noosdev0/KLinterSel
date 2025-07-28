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
from scipy.special import rel_entr
from scipy.stats import chi2


# Configurar el logging para escribir mensajes en un archivo
logging.basicConfig(filename='memory.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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

UNIFORME=False
KILOBASES=False
FIGURE=False
STATS=False

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

    # Si la memoria es suficiente, continúa con el procesamiento
    #print(f"Available memory {memoria_disponible_gib}, requiring {memoria_requerida_gib}. Procesing block...")



def procesar_replica(args):
    r, numfiles, datos_filtrados, Lsitios, c, maxsites = args
    rmuestra = []
    for i in range(1, numfiles):
        rmuestra.append(np.array(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)))
    return totdist(*rmuestra, end=maxsites)

##def calcular_avertot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites,maxproc=MAXPROC):
##    '''
##    Usage: avertot_distances = calcular_avertot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites)
##    '''
##    avertot_distances = np.zeros(maxsites)
##
##    # Crear una lista de argumentos para cada réplica
##    args_list = [(r, numfiles, datos_filtrados, Lsitios, c, maxsites) for r in range(numrs)]
##
##    # Usar multiprocessing para paralelizar
##    with Pool(processes=maxproc) as pool:
##        results = pool.map(procesar_replica, args_list)
##
##    # Sumar todos los resultados y calcular la media
##    for result in results:
##        avertot_distances += result
##    avertot_distances /= numrs
##
##    return avertot_distances

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
    Usage: arrP[c] = calcular_arrP_paralelo_bloques(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=num_processes,block_size=BLOCK_SIZE)
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

            # Define el objeto de la ruta para poder separar la ruta y el nombre del fichero
            path = Path(ruta)

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
        # Si no es un número, devolver el valor original
        #print(f'Excepción de ValueError leyendo {path.name} en {eliminar_comas_csv.__name__}')
        print(ve)
        return False

# Función para eliminar las comas de los números en un csv. Devuelve el nuevo nombre de fichero sin comas en los números
# Si no había comas devuelve el nombre del fichero original
def eliminar_comas_csv(ruta,delimiter, debug=False):
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

            # Define el objeto de la ruta para poder separar la ruta y el nombre del fichero
            path = Path(ruta)

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
        return False
    
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

    #open and manage the csv file
    #https://www.w3resource.com/python-exercises/tkinter/python-tkinter-dialogs-and-file-handling-exercise-9.php
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
        #lista_filtrada = [datos_agrupados[clave] for clave in sorted(datos_agrupados.keys())]

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
                
        #
            return filtered_ndarrays, totsnps
        else:
            return ndarrays_filtrados, totsnps
    
    except ValueError as valerr:
        print(valerr)
        return False


def filter_crompos(ruta, cromcol=0, poscol=1, Kmax=np.inf, D=-np.inf, header=True):
    """ Abre un fichero y almacena en un array de numpy la columna con el identificador de cromosoma y la de la posición
        
        Returns: Una lista de arrays cada elemento (array) de la lista corresponde a un cromosoma y dentro de cada array están las posiciones
        Si el número de posiciones es mayor de Kmax entonces identifica clusters de posiciones que estén todos dentro de la distancia D
        y se queda solo con los extremos
        
    """

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
        #lista_filtrada = [datos_agrupados[clave] for clave in sorted(datos_agrupados.keys())]

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
        return False


def filter_norm(ruta,cromid=1, poscol=1, critcol=-1, criter=1, Kmax=np.inf, D=-np.inf):
    """ Abre un fichero y almacena en un array de numpy las posiciones cuya columna critcol cumplen el criterio criter (>=criter)
        El formato del fichero .norm debe ser
        id	pos	gpos	p1	ihh1	p2	ihh2	xpehh	normxpehh	crit
        Se asume un único cromosoma y por defecto se filtran las posiciones que tengan valor >=criter en la última columna (critcol=-1)
        Returns:
        Una lista de 1 array que corresponde al único cromosoma y los elementos del array son  las posiciones que tenían valor crit>=criter
    """

    try:

        # Diccionario para agrupar los valores de la columna poscol según los valores de la columna cromcol:
        # en el caso de norm solo hay un cromosoma pero se deja así para que pueda extenderse a más cromosomas
        datos_agrupados = defaultdict(list)
       
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
        return False


def mkdir(ruta):
    """ si no existe crea el directorio en la ruta indicada. Devuelve la ruta
        completa al directorio
    """
    ruta=ruta.strip() # limpia los espacios en blanco al inicio y fin

    path = Path(ruta)

    if(path.is_dir()): #ya existe
        return ruta
    elif(path.is_file()):
        print(f'Ya existe un fichero llamado {ruta} no se creará el directorio')
        return ""        
    # Crea todos los directorios parentales necesarios si no existen, si el directorio ya existe no lanza error
    path.mkdir(parents=True, exist_ok=True) #Requiere from pathlib import Path

    return ruta

def RelEntr(arr1,arr2):

    assert len(arr1) == len(arr2), "Arrays deben tener la misma longitud"
    
    # Normalizar los arrays
    P = arr1 / arr1.sum()
    Q = arr2 / arr2.sum()

    # Calcular la entropia relativa KL
    rel_entr_values = rel_entr(P, Q) # Cantidad de información que se pierde cuando Q se usa para aproximar P
    
    kl_div = np.sum(np.where(Q == 0, 0, rel_entr_values)) # corrige valores inf causados por Q==0

    
    # Tamaño de la muestra 
    #n = len(arr1)

    # Estadística de prueba
    #test_statistic = 2 * n * kl_div

    # Grados de libertad
    #df = len(P) - 1

    # Valor p
    #p_value = 1 - chi2.cdf(test_statistic, df)

    return kl_div

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
        np.percentile(rtot_distances[r][0], 50) <= emedian[c]
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
                #  usamos extend() para añadir cada una de las diferencias calculadas como elementos individuales a la lista all_differences
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

def RelEntr_vectorized(arr1_3d, arr2):
    """
    Calcula la divergencia KL entre cada array en una matriz 3D y un array de referencia.

    Parámetros:
    - arr1_3d: Matriz 3D de forma (numrs, longitud_array)
    - arr2: Array 1D que se usará como referencia para cada array en arr1_3d.

    Retorna:
    - kl_divs: Array de divergencias KL para cada conjunto de datos en arr1_3d.

    Usage: kl_divs_result = RelEntr_vectorized(rtot_array_3d, avertot_distances)
    """
    # Convertir la lista de ndarrays a un solo ndarray 3D
    rtot_array = np.array(arr1_3d)
    # Verificar que las dimensiones coinciden
    assert rtot_array.shape[1] == arr2.shape[0], "Los arrays deben tener longitudes compatibles"

    # Normalizar arr1 y arr2 para cada conjunto de datos a lo largo de la primera dimensión
    P = rtot_array / rtot_array.sum(axis=1, keepdims=True)
    Q = arr2 / arr2.sum()

    # Asegúrate de que Q es un array 2D para las operaciones broadcast
    Q_expanded = np.expand_dims(Q, axis=0)  # Cambiar forma para broadcast (1, longitud_array)

    # Calcular la divergencia KL para cada par de arrays
    rel_entr_values = rel_entr(P, Q_expanded)
    kl_divs = np.sum(np.where(Q_expanded == 0, 0, rel_entr_values), axis=1)

    return kl_divs

def RelEntr_vectorized_blocked(arr1_3d, arr2, block_size=100):
    # Convertir la lista de ndarrays a un solo ndarray 3D
    rtot_array = np.array(arr1_3d)
    num_samples = rtot_array.shape[0]
    kl_divs = np.zeros(num_samples)

    for i in range(0, num_samples, block_size):
        end_idx = min(i + block_size, num_samples)
        block = rtot_array[i:end_idx]
        P = block / block.sum(axis=1, keepdims=True)
        Q = arr2 / arr2.sum()
        Q_expanded = np.expand_dims(Q, axis=0)
        rel_entr_values = rel_entr(P, Q_expanded)
        kl_divs[i:end_idx] = np.sum(np.where(Q_expanded == 0, 0, rel_entr_values), axis=1)

    return kl_divs

def calculate_arrP(rtot_distances, avertot_distances, arrKL, emedian, c, kl_divs):
    '''
    Calcula el valor p dados el array numrs de distancias
    kl_divs = RelEntr_vectorized(rtot_distances, avertot_distances)
    arrP_result = calculate_arrP(rtot_distances, avertot_distances, arrKL, emedian, c, kl_divs)
    '''
    numrs = len(rtot_distances)

    # Calcular el percentil 50 para cada conjunto de datos en rtot_distances
    median_values = np.array([np.percentile(rtot_distances[r][0], 50) for r in range(numrs)])

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
def intersec_Dn_Opt_old(*arrays, D=np.inf, keyname='A_'):
    n_arrays = len(arrays)
    cd = {}

    print(f'Generate all possible combinations of positions between methods that are at a distance <= {D}.')

    for length in range(2, n_arrays + 1):
        print(f'{n_arrays} choose {length}')
        for indices in combinations(range(n_arrays), length):
            indices_sorted = sorted(indices)
            key = keyname + ''.join(map(str, [i + 1 for i in indices_sorted]))
            # Usamos un conjunto para almacenar combinaciones únicas como tuplas
            unique_combinations = set()

            if length == 2:
                i, j = indices_sorted
                array_i, array_j = arrays[i], arrays[j]
                for elem_i in array_i:
                    for elem_j in array_j:
                        if abs(elem_i - elem_j) <= D:
                            combo_tuple = tuple(sorted([elem_i, elem_j]))
                            unique_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in unique_combinations]
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
                                    combo_tuple = tuple(new_combination)
                                    temp_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in temp_combinations]

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
                                                kset.add(elem)
                                    combo_tuple = tuple(new_combination)
                                    temp_combinations.add(combo_tuple)
                cd[key] = [np.array(list(combo)) for combo in temp_combinations]
        
            
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

def main():

    global ALFA
    global numrs  # 
    global RANDOM
    global TEST
    global FIGURE
    global STATS
    
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
                        help='permutations, an integer between 0 and 1E5')

    # Argumento con nombre para hacer un control aleatorio
    parser.add_argument('--rand', action='store_true', default=False,
                        help='control using randomly generated data based on the pattern of the entered files')
    # Argumento con nombre para solo calcular intersecciones de todos los cromosomas (sin hacer test)
    parser.add_argument('--notest', action='store_true', default=False,
                        help='compute intersection without performing KL test')

    # Argumento con nombre para pintar las distribucioens de distancias observada y esperada
    parser.add_argument('--paint', action='store_true', default=False,
                        help='compute observed and expected histogram')

    # Argumento con nombre para pintar las distribucioens de distancias observada y esperada
    parser.add_argument('--stats', action='store_true', default=False,
                        help='compute stats for the given files')
    # Parsear los argumentos
    args = parser.parse_args()


    # Validar el valor de SL
    if not 0 <= args.SL <= 1:
        raise ValueError("The significance level must be between 0 and 1.")
    ALFA=args.SL

    # Validar el valor de dist
    if not -1 <= args.dist <= 1E8:
        raise ValueError("The distance must be an integer between 0 and 1E8.")

    distance=10000 #default value
    if args.dist>=0:
        distance=args.dist
        

    # Validar el valor de perm
    if not 0 <= args.perm <= 100000:
        raise ValueError("The number of permutations must be an integer between 0 and 1E5.")
    elif args.perm!=numrs: #numrs=0 de modo que si no se introduce el argumento o se introduce cualquier valor <>0  el valor se carga en numrs
        numrs = args.perm

    if args.rand==True:
        RANDOM=True

    if args.notest==True:
        TEST=False

    if args.paint==True:
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

        if extname=='csv':
            if(num==0): #El primer fichero no se filtra para el número de sitios
                datos_filtrados.append(FilterCsv(ruta)[0])
            else: #Si hay demasiados sitios significativos (>=Kmax se filtran por clusters de distancia
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

    # Listas para cada fichero (posición 0 es el fichero total, las otras las de los métodos)
    #cada elemento de la lista contiene una lista con el número de sitios totales (primera lista) o significativos por cromosoma
    Lsitios=[]
    for data in datos_filtrados: # Cada elemento de datos filtrados corresponde a los datos de un fichero
        Lsitios.append([fila.size for fila in data])

    numcroms=len(Lsitios[0])

    print(f'The number of chromosomes is {numcroms}')

    if STATS:
        
        for i in range(numfiles):
            print(f"\nSTATS for FILE {nombre[i]}\n")
            for c in range(numcroms):
                print(f"CHROMOSOME {c+1}")
                print("Min:", np.min(datos_filtrados[i][c]))
                print("Max:", np.max(datos_filtrados[i][c]))
                print("Mean:", round(np.mean(datos_filtrados[i][c]),2))
                print("SD:", round(np.std(datos_filtrados[i][c]),2))
                print(f'Median {round(np.percentile(datos_filtrados[i][c],50),2)}')

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

        #print(f'Prob with D12<={int(DClose12)}kb\tD13<={int(DClose13)}kb\tD23<={int(DClose23)}kb')
        print(f'\nCHR\tTOTS\t',end='')
        for i in range(1,numfiles):
            print(f'SEL-{i}\t',end='')
        print(f'KL\tPr\toQ2\teQ2\n')
        
        for c in range(numcroms):
            
            # Calcula el rango para cada cromosoma
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
                        print(f"Error: {e}")
                        print("Please refer to the 'memory.log' file for more details.")
                        sys.exit(1)

                #print(f'The maximum possible number of distances between sites from two methods in the data is {maxsites}')
                
                # Calcula la DISTRIBUCION MEDIA esperada por azar
##                rtot_distances=[]
##                for r in range(numrs):
##                    rmuestra=[]
##
##                    for i in range(1,numfiles):
##
##                        rmuestra.append(np.array(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)))
##
##                    rtot_distances.append(totdist(*rmuestra, end=maxsites))
                avertot_distances=np.zeros(maxsites) # Inicializar
                if not TEMP: # hay suficiente memoria para manejar todas las réplicas
                    # 25 july 25 vectorized improvement
                    rtot_distances =generate_rtot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites)
                    #avertot_distances =avertot_distances//numrs
                    # Apilar los ndarrays a lo largo de un nuevo eje para formar un array 2D
                    #stacked_distances = np.vstack(rtot_distances)
                    # Calcular la media a lo largo del eje 0 (es decir, la media en cada posición)
                    
                    avertot_distances = np.mean(np.vstack(rtot_distances), axis=0)
                    
                else: #NO SE ALMACENAN RÉPLICAS

                    if PARALLEL:
                        print(f'Moving to memory management functions to handle {round(reqram)} GiB')
                        print(f'CALCULATING EXPECTED DISTRIBUTION')
                        avertot_distances = calcular_avertot_distances(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites,maxproc=num_processes)
                    else:

                        for r in range(numrs):
                            #rmuestra=[]

                            #for i in range(1,numfiles):

                                #rmuestra.append(np.array(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)))

                            rmuestra = [
                                np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)
                                for i in range(1, numfiles)
                            ]

                            avertot_distances +=totdist(*rmuestra, end=maxsites) # totdist ya devuelve ordenado el array por eso avertot_distances ya está ordenado
                        avertot_distances/=numrs
                
                if not RANDOM: #DISTRIBUCIÓN OBSERVADA

                    tot_distances[c] = totdist(*data_per_crom, end=maxsites)
                    
                else: #Divergencia entre una distribución al azar y la media
                    #Genera distancias al azar
                    muestras=[]
                    for i in range(1,numfiles):
                        muestras.append(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False))
                        #muestra2.append(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False))

                    tot_distances[c] = totdist(*muestras, end=maxsites)
                    #avertot_distances = totdist(*muestra2, end=maxsites)

                #Calcula la entropía relativa entre la distribución real y la esperada por azar
                arrKL[c] = RelEntr(tot_distances[c],avertot_distances) # Cantidad de información perdida cuando avertot_distances se usa para aproximar tot_distances[c]

                # Calcula la mediana de la distribución de distancias observadas y la de la esperada
                
                omedian[c]=np.percentile(tot_distances[c],50)
                emedian[c]=np.percentile(avertot_distances,50)
                #Q1 = np.percentile(avertot_distances,25)
                #IQR[c]=np.percentile(avertot_distances,75) - Q1
                #limite_inferior = Q1-IQRfactor*IQR[c] # Valor menor 3 veces por debajo del rango intercuartil
                limite_inferior = avertot_distances[0]

                if omedian[c]<= emedian[c]:

                    '''
                    - REMUESTREO:

                    '''
                        
                # Generar las muestras sin reemplazo dentro de cada muestra
                #np.random.seed(42)
                                   

                    if UNIFORME: # muestrea posiciones de una uniforme

                        for _ in range(numrs):
                            Umuestra=[]
                            for i in range(1,numfiles):
                                Umuestra.append(np.round(np.random.uniform(size=n1)*rango + min_val, 0))

                            rtot_distances = totdist(*Umuestra, end=maxsites)

                            if RelEntr(tot_distances[c],rtot_distances) <= arrKL[c] and np.percentile(rtot_distances[r][0],50) <= emedian[c]:

                                arrP[c] += 1

                          
                    elif REMUESTREO: # muestrea directamente posiciones reales

                        '''
                        La divergencia de la distribución de distancias entre candidatos con la distribución esperada (media) por azar
                        tiene valor arrKL[c] ¿Cuan probable es obtener un valor igual o mayor que este entre una divergencia obtenida por azar
                        y la esperada por azar?
                        '''
                        #Comento y sustituyo por calculate_vectorized
##                        for r in range(numrs):
##                            Q1=np.percentile(rtot_distances[r],25)
##                            #riqr= np.percentile(rtot_distances[r],75)-Q1
##                            # Cantidad de información perdida cuando avertot_distances se usa para aproximar rtot_distances[r]
##                            if RelEntr(rtot_distances[r],avertot_distances) >= arrKL[c] and np.percentile(rtot_distances[r][0],50) <= emedian[c]: 
##
##                                arrP[c] += 1
##                                
##                               
##
##                        arrP[c]/=numrs
                        if not TEMP: # Las permutaciones están almacenadas en rtot_distances

                            arrP[c] = calculate_vectorized(rtot_distances, avertot_distances, arrKL, emedian, c)
                        else: #Las permutaciones no están almacenadas y se vuelven a hacer para calcular el valor p

                            if PARALLEL:
                                print(f'CALCULATING p-VALUES')
                                #arrP[c] = calcular_arrP_paralelo(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=num_processes)
                                arrP[c] = calcular_arrP_paralelo_bloques(numrs, numfiles, datos_filtrados, Lsitios, c, maxsites, avertot_distances, arrKL, emedian,maxproc=num_processes,block_size=BLOCK_SIZE)
                            else:
                            
                                for r in range(numrs):
                                    #rmuestra=[]

                                    #for i in range(1,numfiles):

                                        #rmuestra.append(np.array(np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)))

                                    rmuestra = [
                                        np.random.choice(datos_filtrados[0][c], size=Lsitios[i][c], replace=False)
                                        for i in range(1, numfiles)
                                    ]

                                    rtot_distances = totdist(*rmuestra, end=maxsites)

                                    if RelEntr(rtot_distances,avertot_distances) >= arrKL[c] and np.percentile(rtot_distances[0],50) <= emedian[c]: 
                                    
                                        arrP[c] +=1

                                arrP[c]/=numrs

                        
                        #kl_divs = RelEntr_vectorized(rtot_distances, avertot_distances)
                        #arrP[c] = calculate_arrP(rtot_distances, avertot_distances, arrKL, emedian, c, kl_divs)

                        
                        #kl_divs = RelEntr_vectorized(rtot_distances, avertot_distances)

                        #forma_array = (numrs, maxsites)

                        #tamano_bloque = calcular_tamano_bloque(forma_array,factor_seguridad=factor_seguridad)

                        #kl_divs = RelEntr_vectorized_blocked(rtot_distances, avertot_distances, block_size=tamano_bloque)
                        
                        #arrP[c] = calculate_arrP(rtot_distances, avertot_distances, arrKL, emedian, c, kl_divs)

                        #assert arrP[c] == arrP_result, f"{arrP[c]} distinto de {arrP_result}"
                        
                    else:

                        arrP[c]=1

                else: # Si la mediana observada no es menor que la esperada
                    arrP[c]=1
            else: # No hay candidatos para este cromosoma
                
                arrP[c]=1

            print(f'{c+1}\t{Lsitios[0][c]}\t',end='')
            for i in range(1,numfiles):
                print(f'{Lsitios[i][c]}\t',end='')
            print(f'{np.format_float_positional(arrKL[c], precision=4)}\t{arrP[c]}\t{omedian[c]}\t{emedian[c]}\n')

            if FIGURE: # tot_distances[c],avertot_distances
                # Crear una figura con dos subplots
##                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
##                # Graficar el primer array en el primer subplot usando kdeplot
##                sns.kdeplot(tot_distances[c], ax=axes[0])
##                axes[0].set_title('OBSERVED')
##                
##
##                # Graficar el segundo array en el segundo subplot usando kdeplot
##                sns.kdeplot(avertot_distances, ax=axes[1])
##                axes[1].set_title('EXPECTED')

                #nbins=(calculate_bins(tot_distances[c]) + calculate_bins(avertot_distances))//2
                nbins, barwidth = calculate_bins(avertot_distances)
                #print(nbins)
                #sys.exit()
                #nbins= min(1000,int(np.sqrt(len(avertot_distances))))

                # Crear un DataFrame combinado
                df = pd.DataFrame({
                    'value': np.concatenate([tot_distances[c], avertot_distances]),
                    'array': ['OBSERVED'] * len(tot_distances[c]) + ['EXPECTED'] * len(avertot_distances)
                })

                # Calcular el máximo valor entre los dos arrays
                max_value = max(np.max(tot_distances[c]), np.max(avertot_distances))

                

                # Crear el gráfico de distribución con frecuencias
                p = sns.displot(data=df, x='value', col='array', kind='hist',
                                height=5, aspect=2, stat='probability',
                                bins=nbins)  # Ajustar el número de bins según sea necesario

                # Calcular los límites del eje Y para que sean iguales en ambos subplots
                ylims = [ax.get_ylim() for ax in p.axes.flat]
                max_ylim = max([ylim[1] for ylim in ylims])
                
                # Gestionar formato ejes
                for ax in p.axes.flat:
                    for patch in ax.patches:
                        patch.set_facecolor('none')
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1)  # Grosor del borde                    

                    ax.set_xlim(-barwidth, max_value)
                    ax.spines['left'].set_position(('data', -barwidth))
                    ax.set_ylim(0, max_ylim)
                
                    # Configurar cada subplot para mostrar solo las líneas de los ejes X e Y
                    # Eliminar los spines derecho y superior
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)

                    # Mantener solo los spines inferior y izquierdo (ejes X e Y)
                    #ax.spines['bottom'].set_visible(True)
                    #ax.spines['left'].set_visible(True)

                    # Eliminar los títulos de los ejes
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                #Personalizar títulos de cada gráfico
                #titles = ['OBSERVED', 'EXPECTED']
                titles = ['', '']
                for ax, title in zip(p.axes.flat, titles):
                    ax.set_title(title, fontsize=11, fontweight='bold', pad=7)

                #plt.tight_layout()
                plt.show()
                
##                print("Estadísticas para array1:")
##                print("Mínimo:", np.min(tot_distances[c]))
##                print("Máximo:", np.max(tot_distances[c]))
##                print("Media:", np.mean(tot_distances[c]))
##                print("Desviación estándar:", np.std(tot_distances[c]))
##                print(f'Mediana {np.percentile(tot_distances,50)}')
##
##                print("\nEstadísticas para array2:")
##                print("Mínimo:", np.min(avertot_distances))
##                print("Máximo:", np.max(avertot_distances))
##                print("Media:", np.mean(avertot_distances))
##                print("Desviación estándar:", np.std(avertot_distances))
##                print(f'Mediana {np.percentile(avertot_distances,50)}')
    else: # if not TEST
        arrP=np.ones(numcroms)
        ALFA=1.0

    ### Calculamos la INTERSECCIÓN entre métodos solo para los cromosomas significativos o
    # si queremos para todos los cromosomas basta poner ALFA=1 o simplemente TEST=False
    #Lo hacemos separado del test KL para tener la opción de no hacer el KL con el argumento notest
    keyname='SEL_'
    if not RANDOM and not FIGURE: #Tanto la opción --rand como la opción --paint inactivan el cálculo de intersecciones
        
        print('\n### INTERSECTION ANALYSIS ###')
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
                
                #iniDic=create_combination_dict(numfiles-1,keyname=keyname) #hay un fichero de datos originales (no se cuenta) y los otros uno por método son lo que interseccionan
                #for item in iniDic.items():
                    #print(item)

                #cd_test[c] =intersec_Dn(*data_per_crom, D=distance,keyname=keyname)
                cd[c], kset[c] =intersec_Dn_Opt_no_sort(*data_per_crom, D=distance,keyname=keyname)
                

    ##            if compare_diccionaries(cd_test[c],cd[c]):
    ##                print(f"Ambas funciones producen los mismos resultados para el cromosoma {c+1}.")
    ##            else:
    ##                print(f"Las funciones producen resultados diferentes para el cromosoma {c+1}.")
    ##                sys.exit()
                

            
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
    out_name='KLiS'
    out_name+='_Kmax_' +str(args.Kmax)+ '_D'+formatted_distance+'_'+moment
    samefiles=True

    for i in range(1,numfiles-1):

        if nombre[i].upper()!=nombre[i+1].upper():
            samefiles=False
            break

    if samefiles:    
        out_name+='_SAMEFILES'

    out_name2 = out_name
    
    if UNIFORME:
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
                f.write(f'{np.format_float_positional(arrKL[c], precision=4)}{delimiter}{arrP[c]}{delimiter}{omedian[c]}{delimiter}{emedian[c]}\n')

    # Name for the interseccion file
    if not RANDOM:
        text= 'INTERSEC'
        out_name2= out_name2.replace('KLiS',text)+'.tsv'
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
    #if FIGURE:
        #mkdir(dirFigs)
        #myutils.creafig(datos_filtrados_0,datos_filtrados_1,datos_filtrados_2,arr1name,arr2name, arr3name,XLABEL='Posiciones',YLABEL='',YTICK='CHR',save=True,ruta=dirFigs)
        #myutils.creafig_multi(datos_filtrados_0,arr1name,[datos_filtrados_1,datos_filtrados_2],[arr2name, arr3name],XLABEL='Posiciones',YLABEL='',YTICK='CHR',save=True,ruta=dirFigs)

if __name__ == "__main__":
    main()
