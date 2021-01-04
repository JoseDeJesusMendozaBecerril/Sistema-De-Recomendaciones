import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import sklearn





#DATA FRAME COMPLETO
Df = pd.read_csv('USvideos.csv')
print("\n**** El numero de renglones de mi dataframe: " , len(Df.index))





#PORCION DEL DATAFRAME
df = Df.sample(frac=0.001)              #Tomar solo el 1 porciento de registros
df = df.reset_index(drop=True)          #Reseteo indice
print("\n**** Numero de renglones y columnas de mi dataframe: " , df.shape)                         #Muestra las filas y columnas del dataframe





#LIMPIEZA DE LOS DATOS
print("\n**** Estos son los renglones de los tags es decir los registros")
print(df['tags'].head())                # Imprime los renglones del Tags


print("\n**** Estos son los renglones de los tags despues de la limpieza")
df['ntags'] = df['tags'].str.replace('[{}]'.format(string.punctuation) , ' ') # Remplaza los corchetes y llaves por un espacio vacio para limpiar los datos de la API
print(df['ntags'].head())                




#SIMILITUD COSENO PARA ENCONTRAR LA FRECUENCIA DE LAS PALABRAS Y ENCONTRAR LOS MAS RECOMENDADOS O PARECIDOS

tfidf = TfidfVectorizer(stop_words='english')  # descarto the a an etc
df['ntags'] = df['ntags'] .fillna('') 

#MATRIZ 
#TRANSFORMO LOS DATOS A NUMEROS
tfidf_matrix = tfidf.fit_transform(df['ntags'])
print(tfidf.vocabulary_) #Palabras y cuantas veces se repiten
print("\n****Numero de videos , Palabras totales para describir los videos: " , tfidf_matrix.shape) #cuantos videos y cuantas palabras para poder describir esos videos


#SIMILITUD COSENO
cosine_sim = linear_kernel(tfidf_matrix , tfidf_matrix) #Comparando la matriz que me va permitir cuando llame una pelicula y seleccionar las peliculas que tienen palabras relacionadas con ella
indices = pd.Series(df.index , index=df['title']).drop_duplicates()


def obten_recomendacion(title,cosine_sim=cosine_sim):
    idx=indices[title]
    #puntuacion dependiendo de la singularidad coseno
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores) con lambda selecciono posicion 1 que es el argumento que toma en cuenta para ordenar
    sim_scores = sorted(sim_scores , key=lambda x:x[1] , reverse = True) #Esto solo es para ordenar de mayor a menor puntuacion o similitud
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores] #relacionar cada uno de los puntajes con los indices de la pelicula

    return df['title'].iloc[movie_indices]

titulo = str(df.iloc[3]['title'])

print("-> Tu seleccionaste: " + titulo)
print("-> Tus recomendaciones son: ")
print(obten_recomendacion(titulo))
