##############################################################################################
# App de Reglas de asociacion en textos (palabras relacionadas)
##############################################################################################

#**************************************************************************************************************
# [A] Importar LIbrerias a Utilizar
#**************************************************************************************************************

import streamlit as st

import pandas as pd
import numpy as np

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from apyori import apriori
import re
from unicodedata import normalize

import warnings
warnings.filterwarnings('ignore')



#**************************************************************************************************************
# [B] Crear funciones utiles para posterior uso
#**************************************************************************************************************

#**************************************************************************************************************
# B.1 Funcion de reglas apriori a df

@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def inspect(output):
  lhs        = [tuple(result[2][0][0])[0] for result in output]
  rhs        = [tuple(result[2][0][1])[0] for result in output]
  support    = [result[1] for result in output]
  confidence = [result[2][0][2] for result in output]
  lift       = [result[2][0][3] for result in output]
  return list(zip(lhs, rhs, support, confidence, lift))


#**************************************************************************************************************
# B.2 Funcion de entregables df-apriori cuando input es texto

@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def apriori_texto(
  df, # dataframe donde esta la data
  columna_item, # nombre de la columna donde va el texto
  min_support = 0.01, # parametros propios de apriori 
  min_confidence = 0.01, # parametros propios de apriori 
  min_lift = 3, # parametros propios de apriori 
  min_length = 2, # parametros propios de apriori 
  max_length = 2 # parametros propios de apriori 
):

  # crear un id correlativo 
  df['ID']=range(1,1+len(df))

  # seleccionar solo columnas relevantes y cambiar nombre
  df = df[['ID',columna_item]].rename(columns={columna_item:'text'})

  # separar df palabra por palabra 
  df_palabras = df[['ID','text']].assign(
    palabra=df['text'].str.split()
    ).explode('palabra')

  # eliminar campo de texto (mensaje)
  df_palabras2 = df_palabras.drop('text', axis = 1) 

  # limpiar palabra 
  df_palabras2['palabra2']=df_palabras2['palabra'].apply(lambda x:
    normalize(
      'NFC',
      re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
        r"\1",
        normalize(
          'NFD',
          re.sub(r'[^\w\s]', '', x.lower().strip())
          ),
        0,
        re.I
        )
      )
    )

  # leer stopwords y pasar a lista
  stop_words = pd.read_csv('stop_words_spanish.txt', sep=' ', header=None,encoding='latin-1')
  stop_words2 = stop_words.iloc[:,0].tolist()

  # crear marca de stopword
  df_palabras2['StopWord']=df_palabras2['palabra'].apply(lambda x: 
    1 if 
    x in stop_words2
    or len(x)<4 
    or len(x)>13 
    or len(re.sub('j|a', '',x))==0 # quitar todos los jajajajajaja
    else 0
    )

  # quedarse con df quitando stopwords
  df_palabras3=df_palabras2.loc[df_palabras2['StopWord']==0,['ID','palabra2']]

  # Crear lista para apriori https://stackoverflow.com/questions/62270442/how-to-convert-a-dataframe-into-the-dataframe-for-apriori-algorithm
  lista_apriori = df_palabras3.groupby('ID')['palabra2'].apply(list).values

  # arrojar regla de priori 
  reglas = apriori(
    transactions = lista_apriori, 
    min_support = min_support, 
    min_confidence = min_confidence, 
    min_lift = min_lift, 
    min_length = min_length, 
    max_length = max_length
    )

  # volcar entregable en un df (usar funcion pre_construida)
  df_reglas = pd.DataFrame(
    inspect(list(reglas)), # usar funcion propia "inspect" creada en el comienzo
    columns = ['Antecedente', 'Consecuente', 'Soporte', 'Confianza', 'Lift']
    ).sort_values(by='Confianza', ascending=False).reset_index(drop=True)


  # Calcular tabla de frecuencias
  df_freq_0 = df_palabras3.groupby(['palabra2']).agg( 
      Conteo = pd.NamedAgg(column = 'palabra2', aggfunc = len)
  ).reset_index() 

  # calcular valor del peso
  df_freq_0['Porc']=df_freq_0['Conteo']/np.sum(df_freq_0['Conteo'])

  # ordenar y cambiar nombres
  df_freq_0 = df_freq_0.sort_values(
    by='Conteo', ascending=False
    ).reset_index(drop=True).rename(columns = {'palabra2':'Item'})
  
  # retornar entregables 
  return df_reglas,df_freq_0



#**************************************************************************************************************
# B.3 Funcion de entregables df-apriori cuando input es transacciones


@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def apriori_trxs(
  df, # dataframe donde esta la data
  columna_id, # nombre de la columna donde va el identidicador registro
  columna_item, # nombre de la columna donde va el item comprado
  min_support = 0.001, # parametros propios de apriori 
  min_confidence = 0.001, # parametros propios de apriori 
  min_lift = 3, # parametros propios de apriori 
  min_length = 2, # parametros propios de apriori 
  max_length = 2 # parametros propios de apriori 
):
 
  # crear lista de elementos para apriori 
  lista_apriori = df.groupby(columna_id)[columna_item].agg(list).values.tolist()

  # arrojar regla de priori 
  reglas = apriori(
    transactions = lista_apriori, 
    min_support = min_support, 
    min_confidence = min_confidence, 
    min_lift = min_lift, 
    min_length = min_length, 
    max_length = max_length
    )

  # volcar entregable en un df
  df_reglas = pd.DataFrame(
    inspect(list(reglas)), # usar funcion propia "inspect" creada en el comienzo
    columns = ['Antecedente', 'Consecuente', 'Soporte', 'Confianza', 'Lift']
    ).sort_values(by='Confianza', ascending=False).reset_index(drop=True)


  # Calcular tabla de frecuencias
  df_freq_0 = df.groupby([columna_item]).agg( 
      Conteo = pd.NamedAgg(column = columna_item, aggfunc = len)
  ).reset_index() 

  # calcular valor del peso
  df_freq_0['Porc']=df_freq_0['Conteo']/np.sum(df_freq_0['Conteo'])

  # ordenar y cambiar nombres
  df_freq_0 = df_freq_0.sort_values(by='Conteo', ascending=False).reset_index(drop=True)
  df_freq_0 = df_freq_0.rename(columns = {columna_item:'Item'})

  # retornar entregables 
  return df_reglas,df_freq_0



#**************************************************************************************************************
# B.4 Funcion de entregables graficos-apriori 



@st.cache(suppress_st_warning=True,allow_output_mutation=True) # https://docs.streamlit.io/library/advanced-features/caching
def apriori_graficos(
  df_reglas, # df de reglas ordenado segun confianza (u otro indicador relevante)
  top_reglas, # cantidad de reglas a seleccionar
  df_freq_0, # df de frecuencias de totalidad de items (sacado de df original)
):

  #__________________________________________________________________
  # crear df de reglas segun top ingresado
  df_reglas2 = df_reglas.head(top_reglas)

  # definir listado de colores a utilizar
  lista_colores1 = px.colors.sequential.Blues[2:] # px.colors.sequential.swatches_continuous()

  # determinar indice en listado de colores a usar
  df_reglas2['numero_color']=np.interp(
    df_reglas2['Lift'],
    [min(df_reglas2['Lift']),max(df_reglas2['Lift'])], 
    [0,len(lista_colores1)-1]
    )

  # asignar color segun indice anterior
  df_reglas2['color'] = df_reglas2['numero_color'].apply(
    lambda x: lista_colores1[int(np.floor(x))]
  )

  # determinar el ancho a utilizar segun algun indicador
  df_reglas2['ancho']=np.interp(
    df_reglas2['Confianza'],
    [min(df_reglas2['Confianza']),max(df_reglas2['Confianza'])], 
    [1,4]
    )
  
  #__________________________________________________________________
  # rescatar listado de items a utilizar segun top de reglas seleccionadas 
  listado_items = list(
    np.unique(
      list(df_reglas2['Antecedente'])+list(df_reglas2['Consecuente'])
      )
    )

  # determinar df de frecuencias a utilizar
  df_freq =  df_freq_0.loc[
    df_freq_0['Item'].isin(listado_items)
    ].sort_values(by='Conteo', ascending=False).reset_index()

  # determinar size segun conteo
  df_freq['Tamano']=np.interp(
    df_freq['Conteo'],
    [min(df_freq['Conteo']),max(df_freq['Conteo'])], 
    [15,30]
    )

  # asignar cantidad de conexiones de ese item 
  df_freq = pd.merge(
    df_freq,
    df_reglas2.groupby(['Antecedente']).agg(
      N_Conexiones = pd.NamedAgg(column = 'Consecuente', aggfunc = len)
      ).reset_index(),
    how='left',
    left_on='Item',
    right_on='Antecedente'
  )
  df_freq['N_Conexiones'] = df_freq['N_Conexiones'].fillna(0)

  # definir lista de colores a usar 
  lista_colores2 = px.colors.sequential.BuGn[4:] # px.colors.sequential.swatches_continuous()

  # asignar numero de color en listado de colores definido antes
  df_freq['numero_color']=np.interp(
    df_freq['N_Conexiones'],
    [min(df_freq['N_Conexiones']),max(df_freq['N_Conexiones'])], 
    [0,len(lista_colores2)-1]
    )

  # asignar colores
  df_freq['color'] = df_freq['numero_color'].apply(
    lambda x: lista_colores2[int(np.floor(x))]
  )

  #__________________________________________________________________
  # crear objeto de grafo
  Grafo = nx.Graph()

  # crear listado de nodos
  lista_nodos = list(df_freq['Item'])

  # Agregar nodos
  Grafo.add_nodes_from(lista_nodos)
  # Grafo.nodes()

  # crear listado de aristas
  lista_aristas = df_reglas2[['Antecedente','Consecuente']].values.tolist()

  # agregar aristas (desafortunadamente no se agregan en el orden deseado)
  Grafo.add_edges_from(lista_aristas) # innecesario este paso
  # Grafo.edges()

  # crear una lista de posiciones
  pos = nx.spring_layout(Grafo)

  # remapear posiciones sobre valores positivos 
  pos2 = {}
  for i in range(len(pos)):
    pos2[list(pos.keys())[i]]=np.array([
      list(pos.values())[i][0]+100,
      list(pos.values())[i][1]+100
      ])

  #__________________________________________________________________
  # Con las nuevas posiciones, determinar cuantos enlaces tiene por cuadrante
  df_reglas2p = df_reglas2[['Antecedente','Consecuente']]
  df_reglas2p['x_a']=df_reglas2p['Antecedente'].apply(lambda x: pos2[x][0])
  df_reglas2p['y_a']=df_reglas2p['Antecedente'].apply(lambda x: pos2[x][1])
  df_reglas2p['x_c']=df_reglas2p['Consecuente'].apply(lambda x: pos2[x][0])
  df_reglas2p['y_c']=df_reglas2p['Consecuente'].apply(lambda x: pos2[x][1])

  df_reglas2p['a_c']=df_reglas2p.apply(lambda p:
    'top left' if p['y_c']>=p['y_a'] and p['x_a']>=p['x_c'] else 
    'top right' if p['y_c']>=p['y_a'] and p['x_a']<=p['x_c'] else 
    'bottom left' if p['y_c']<=p['y_a'] and p['x_a']>=p['x_c'] else 
    'bottom right',
    axis=1
    )

  df_reglas2p['c_a']=df_reglas2p['a_c'].apply(lambda p:
    'top left' if p=='bottom right' else 
    'top right' if p=='bottom left' else 
    'bottom left' if p=='top right' else 
    'bottom right'
    )

  # determinar ubicacion de posterior etiqueta donde hayan menos enlaces
  df_reglas2p2 = pd.concat([
    df_reglas2p[['Antecedente','a_c']].rename(
      columns = {'Antecedente':'Item','a_c':'enlaces'}
      ),
    df_reglas2p[['Consecuente','c_a']].rename(
      columns = {'Consecuente':'Item','c_a':'enlaces'}
      )
    ]).groupby(['Item','enlaces']).agg(
      N = pd.NamedAgg(column = 'Item', aggfunc = len)
      ).reset_index().pivot_table(
        index='Item', 
        columns='enlaces', 
        values='N', 
        aggfunc=np.sum,
        fill_value=0
        ).reset_index()


  df_reglas2p2['ubicacion'] = df_reglas2p2[
    ['bottom left','bottom right','top left','top right']
    ].idxmin(axis=1)


  df_freq2 = pd.merge(
    df_freq,
    df_reglas2p2[['Item','ubicacion']],
    how='left',
    left_on='Item',
    right_on='Item'
  )
  df_freq2['ubicacion'] = df_freq2['ubicacion'].fillna('top right')

  #__________________________________________________________________
  # Crear Objeto Grafico
  fig = go.Figure()

  # Agregar Nodos 
  for nodo in Grafo.nodes():
    
    x_nodo, y_nodo = pos2[nodo]
    
    filtro_df = df_freq2['Item'] == nodo
    
    tamano = df_freq2.loc[filtro_df, 'Tamano'].iloc[0]
    color = df_freq2.loc[filtro_df, 'color'].iloc[0]
    
    cantidad = str(df_freq2.loc[filtro_df, 'Conteo'].iloc[0])
    porcentaje = str(round(100*df_freq2.loc[filtro_df, 'Porc'].iloc[0],2))+'%'
    conexiones = str(int(df_freq2.loc[filtro_df, 'N_Conexiones'].iloc[0]))
    
    ubicacion = df_freq2.loc[filtro_df, 'ubicacion'].iloc[0]
    
    texto = '<br>Frecuencia: '+cantidad+'<br>Porcentaje: '+porcentaje+'<br>Conexiones: '+conexiones
    
    fig.add_trace(go.Scatter(
      x=[x_nodo], 
      y=[y_nodo], 
      mode='markers+text',
      marker=dict(size=tamano,color=color), 
      name=nodo,
      showlegend=True,
      hovertemplate = nodo + texto,
      text = nodo,
      textposition = ubicacion,
      ))

  # Agregar Aristas  
  lista_flechas = []
  for arista in lista_aristas:
    
    x0_i, y0_i = pos2[arista[0]]
    x1_i, y1_i = pos2[arista[1]]
    
    m = (y1_i-y0_i)/(x1_i-x0_i)
    n = y1_i - m*x1_i
      
    # mejor probar que la variacion de x sea un porcentaje sobre este dependiendo del valor de la pendiente
    factor_flecha = np.interp(np.arctan(abs(m)),[0,3.1416/2],[0.040,0.030])

    if x0_i>x1_i:
      x0 = x0_i - abs(x0_i-x1_i)*factor_flecha
      x1 = x1_i + abs(x0_i-x1_i)*factor_flecha
      y0 = m*x0+n
      y1 = m*x1+n
    else:
      x0 = x0_i + abs(x0_i-x1_i)*factor_flecha
      x1 = x1_i - abs(x0_i-x1_i)*factor_flecha
      y0 = m*x0+n
      y1 = m*x1+n
          
    filtro_df = (
      (df_reglas2['Antecedente'] == arista[0]) & 
      (df_reglas2['Consecuente'] == arista[1])
      ) 
    
    Color = df_reglas2.loc[filtro_df,'color'].iloc[0] 
    Ancho = df_reglas2.loc[filtro_df,'ancho'].iloc[0] 
    
    lista_flechas.append(go.layout.Annotation(dict(
      x=x1,
      y=y1,
      xref='x', yref='y',
      text='',
      showarrow=True,
      axref='x', ayref='y',
      ax=x0,
      ay=y0,
      arrowhead=2,
      arrowwidth=Ancho,
      arrowcolor=Color
      )))

    Confianza = str(round(df_reglas2.loc[filtro_df,'Confianza'].iloc[0],4))
    Soporte = str(round(df_reglas2.loc[filtro_df,'Soporte'].iloc[0],4))
    Lift = str(round(df_reglas2.loc[filtro_df,'Lift'].iloc[0],4))
    
    texto = arista[0]+' → '+arista[1]+'<br>Confianza: '+Confianza+'<br>Soporte: '+Soporte+'<br>Lift: '+Lift

    x_m = (x0+x1)/2
    y_m = (y0+y1)/2
      
    fig.add_trace(go.Scatter(
      x=[x_m], 
      y=[y_m], 
      mode='markers',
      name='',
      marker=dict(opacity=0,size=10,color=Color), 
      showlegend=False,
      hovertemplate = texto
      ))

  fig.update_layout(
    annotations=lista_flechas,
    showlegend=False, 
    hovermode='closest',
    margin=dict(b=5,l=5,r=5,t=5),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

  #__________________________________________________________________
  # Crear figura de dispersion 

  fig2 = px.scatter(
    df_reglas2,
    x = 'Antecedente',
    y = 'Consecuente',
    size = 'Soporte',
    color = 'Confianza'
    )
  
  fig2.update_layout(
    height=600, 
    width=1200
    )

  #__________________________________________________________________
  # Retornar entregables

  return fig,fig2



#**************************************************************************************************************
# [Z] Comenzar a diseñar App
#**************************************************************************************************************

def main():
  
  # Use the full page instead of a narrow central column
  st.set_page_config(layout='wide')
    
  #=============================================================================================================
  # [01] SideBar
  #=============================================================================================================   

  # titulo inicial 
  st.markdown('## Reglas de Asociacion')
  
  # autoria 
  st.sidebar.markdown('**Autor: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')
      
  # subir archivo 
  Archivo = st.sidebar.file_uploader('Subir Data.csv',type=['csv'])

  #=============================================================================================================
  # [02] Main
  #=============================================================================================================   

  #-------------------------------------------------------------------------------------------------------------
  # [02.1] Main: Parametros 
  #-------------------------------------------------------------------------------------------------------------


  if Archivo is not None:

    # leer archivo (con distintos delimitadores)
    df = pd.read_csv(Archivo,delimiter ='[:|;]',engine='python')
    
    # mostrar data de archivo subido 
    expander1 = st.expander('Data Subida',expanded=True)
    expander1.dataframe(df)
    
    # titulo  
    st.markdown('### 1. Seleccion de Parametros')
      
    # seleccionar tipo de analisis
    col1a,col1b,col1c = st.columns((2,1,1))
 
    tipo_analisis = col1a.radio(
      'Elegir tipo de analisis', 
      ['Texto','Transacciones'], 
      horizontal=True
      )
    
    if tipo_analisis=='Texto':
      
      columna_item = col1b.selectbox(
        'Indicar campo de texto',
        list(df.columns)
        )
    
    else:
      
      columna_id = col1b.selectbox(
        'Indicar campo de identificador',
        list(df.columns)
        )
            
      columna_item = col1c.selectbox(
        'Indicar campo de categoria',
        list(df.columns)
        )
        
    # seleccionar parametros de apriori 
    st.markdown(' ') 
    col2a,col2b,col2c,col2d = st.columns((1,1,1,1))
    
    min_support = col2a.number_input(
      label = 'Soporte Minimo',
      value = 0.001,
      min_value= 0.0001,
      max_value=0.1,
      step=1e-6,
      format="%.5f",
      key=1
    )

    min_confidence = col2b.number_input(
      label = 'Confianza Minima',
      value = 0.001,
      min_value= 0.0001,
      max_value=0.1,
      step=1e-6,
      format="%.5f",
      key=2
    )    
    
    top_reglas = col2c.number_input(
      label = 'Top reglas en grafico',
      value = 25,
      min_value= 5,
      max_value=100,
      key=3
    )   
    
    col2d.write('')
    col2d.write('')
    Boton_ejecutar = col2d.button(label='Ejecutar')
    
    
  #-------------------------------------------------------------------------------------------------------------
  # [02.2] Main: Entregables - calcular
  #-------------------------------------------------------------------------------------------------------------

    # truco para no alterar resultados desde cero apretando boton
    if st.session_state.get('button') != True:
      st.session_state['button'] = Boton_ejecutar

    if st.session_state['button']:
      
      # calcular entregables segun parametros ingresados
      if tipo_analisis=='Texto':
        
        df_reglas,df_freqs = apriori_texto(
          df = df, # dataframe donde esta la data
          columna_item = columna_item, # nombre de la columna donde va el texto
          min_support = min_support, # parametros propios de apriori 
          min_confidence = min_confidence, # parametros propios de apriori 
          min_lift = 3, # parametros propios de apriori 
          min_length = 2, # parametros propios de apriori 
          max_length = 2 # parametros propios de apriori 
          )       
      else:
        
        df_reglas,df_freqs = apriori_trxs(
          df = df, # dataframe donde esta la data
          columna_id = columna_id, # nombre de la columna donde va el identidicador registro
          columna_item = columna_item, # nombre de la columna donde va el texto
          min_support = min_support, # parametros propios de apriori 
          min_confidence = min_confidence, # parametros propios de apriori 
          min_lift = 3, # parametros propios de apriori 
          min_length = 2, # parametros propios de apriori 
          max_length = 2 # parametros propios de apriori 
          )      
        
      # calcular graficos asociados
      graf1,graf2 = apriori_graficos(
        df_reglas = df_reglas, # df de reglas ordenado segun confianza (u otro indicador relevante)
        top_reglas = top_reglas, # cantidad de reglas a seleccionar
        df_freq_0 = df_freqs # df de frecuencias de totalidad de items (sacado de df original)
        )
            

  #-------------------------------------------------------------------------------------------------------------
  # [02.3] Main: Entregables - mostrar
  #-------------------------------------------------------------------------------------------------------------


      # titulo  
      st.markdown('### 2. Entregables') 
      
      expander2 = st.expander('Reglas de Asociacion',expanded=True)
      expander2.download_button(
        'Descargar tabla',
        df_reglas.to_csv().encode('utf-8'),
        'Reglas Asociacion.csv',
        'text/csv',
        key='download-csv'
        )
      expander2.dataframe(df_reglas)
        
      expander3 = st.expander('Grafo de Reglas',expanded=True)
      expander3.plotly_chart(graf1, use_container_width=True)
      expander3.write('Color Nodo: Cantidad de Conexiones del Item')
      expander3.write('Tamaño Nodo: Frecuencia del Item')
      expander3.write('Color Flecha: Lift de la regla')
      expander3.write('Grosor Flecha: Confianza de la regla')
      
      
      
      expander4 = st.expander('Antecedente vs Consecuente',expanded=True)
      expander4.plotly_chart(graf2, use_container_width=True)
      expander4.write('Color: Confianza de la regla')
      expander4.write('Tamaño: Soporte de la regla')
    

# arrojar main para lanzar App
if __name__=='__main__':
    main()
    
# Escribir en terminal: streamlit run App_Apriori4.py
# !streamlit run App_Apriori4.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/15_Streamlit App de Reglas de Asociacion/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit

