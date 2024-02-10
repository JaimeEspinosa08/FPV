#!/usr/bin/env python
# coding: utf-8

# In[351]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
df = pd.read_excel('MDT.xlsx', sheet_name='MDT')


# In[310]:


df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M:%S').dt.time


# In[311]:


df.info()


# In[312]:


numeric_df = df.select_dtypes(include=['float64'])

# Matriz de correlación
correlation_matrix = numeric_df.corr()
potencia_relation = correlation_matrix['POWER'].sort_values(ascending=False)
print(potencia_relation)



# In[313]:


# Hacer indice al tiempo
df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' +
                                df['TIME'].astype(str))
df.set_index('DATETIME', inplace=True)

sorted_df = df.sort_values(by='POWER', ascending=False)

# Tiempos donde la potencia es el mejor 10%
top_10_percent = sorted_df['POWER'].quantile(0.75)
top_times = sorted_df[sorted_df['POWER'] >= top_10_percent].index.time

# Tiempos comunes mostrados de menor a mayor y por coincidencias

print(top_10_percent)
print(top_times.min(),top_times.max())
len(top_times)



# In[314]:


# Hacer indice al tiempo
df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' +
                                df['TIME'].astype(str))
df.set_index('DATETIME', inplace=True)

sorted_df = df.sort_values(by='IRRADIANCE', ascending=False)

# Tiempos donde la potencia es el mejor 10%
top_10_percent = sorted_df['IRRADIANCE'].quantile(0.9)
top_times = sorted_df[sorted_df['IRRADIANCE'] >= top_10_percent].index.time

# Tiempos comunes mostrados de menor a mayor y por coincidencias

print(top_10_percent)
print(top_times.min(),top_times.max())



# In[363]:


# Calculo Q3
q1_potencia = df['POWER'].quantile(0.25)
q2_potencia = df['POWER'].quantile(0.50)
q3_potencia = df['POWER'].quantile(0.75)
m_potencia = df['POWER'].mean()

m_temp = df['TEMP PANEL'].mean()


#Cuartiles Eficiecia
q1_eff = df['EFFICIENCY'].quantile(0.25)
q2_eff = df['EFFICIENCY'].quantile(0.50)
q3_eff = df['EFFICIENCY'].quantile(0.75)
m_eff = df['EFFICIENCY'].mean()

#Cuartiles IRRADIANCE
q1_irr = df['IRRADIANCE'].quantile(0.25)
q2_irr = df['IRRADIANCE'].quantile(0.50)
q3_irr = df['IRRADIANCE'].quantile(0.75)
m_irr = df['IRRADIANCE'].mean()

print(q2_irr)


# In[297]:


import matplotlib.pyplot as plt
import seaborn as sns


# Datos de potencia Q3
df_q3 = df[df['POWER'] >= q3_potencia]

# Plot potencia completa 
plt.figure(figsize=(8, 3))
sns.scatterplot(data=df, x='IRRADIANCE', y='POWER', 
                hue='TEMP PANEL', palette='gnuplot2')

plt.ylabel('POWER')
plt.xlabel('IRRADIANCE')
plt.title('Potencia vs Irradiancia')
plt.show()

# Plot Q3
plt.figure(figsize=(8, 3))
sns.scatterplot(data=df, x=df_q3['IRRADIANCE'], y='POWER', 
                hue='TEMP PANEL', palette='gnuplot2')


plt.ylabel('Q3 POWER')
plt.xlabel('IRRADIANCE')
plt.title('Potencia Q3 vs Irradiancia')
plt.show()


# In[231]:


numeric_df = df_q3.select_dtypes(include=['float64'])

# Matriz de correlación
correlation_matrix = numeric_df.corr()
potencia_relation = correlation_matrix['POWER'].sort_values(ascending=False)
print(potencia_relation)


# In[287]:


# Plot histogram of the 'EFFICIENCY' column
plt.figure(figsize=(8, 3))
sns.histplot(df['EFFICIENCY'], bins=20, kde=False)
plt.title('Histograma de Eficiencia')
plt.xlabel('EFFICIENCY')
plt.ylabel('FREQUENCY')
plt.show()


# In[296]:


fig, ax = plt.subplots(figsize=(5, 3))
ax.boxplot(x=df['EFFICIENCY'],
           patch_artist = True,
           boxprops = dict(facecolor = "lightblue"))
plt.xlabel('Efficiency')
plt.ylabel('Porcentaje %')
plt.show()

print(q1_eff, q2_eff, q3_eff, m_eff)

print(q1_eff-1.5*(q3_eff-q1_eff), q3_eff+1.5*(q3_eff-q1_eff))


# In[307]:


df.columns = [c.strip() for c in df.columns]

# Calculate the correlation matrix
corr_matrix = df[['POWER', 'IRRADIANCE', 'TEMP PANEL','EFFICIENCY']].corr()


# Plot the correlation matrix
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.title('Matriz de correlaciones')
plt.show()



# In[181]:


fig, ax = plt.subplots()
ax.boxplot(x=df['POWER'],
           patch_artist = True,
           boxprops = dict(facecolor = "lightblue"))
plt.xlabel('Potencia')
plt.show()

print(q1_potencia, q2_potencia, q3_potencia, m_potencia)


# In[298]:


# Plot potencia completa 
plt.figure(figsize=(8, 3))
sns.scatterplot(data=df, x='IRRADIANCE', y='EFFICIENCY', 
                hue='TEMP PANEL', palette='gnuplot2')

plt.ylabel('EFFICIENCY')
plt.xlabel('IRRADIANCE')
plt.title('Eficiencia vs Irradiancia')
plt.show()

# Plot potencia completa 

df_irr = df[df['IRRADIANCE'] > q1_irr]
df_irr = df_irr[df_irr['IRRADIANCE'] < q3_irr]

# Plot potencia completa 
plt.figure(figsize=(8, 3))
sns.scatterplot(data=df, x=df_irr['IRRADIANCE'], y='EFFICIENCY', 
                hue='TEMP PANEL', palette='gnuplot2', legend=False)

plt.ylabel('EFFICIENCY')
plt.xlabel('IQR IRRADIANCE')
plt.show()


# In[183]:


# Filter the dataframe for entries where IRRADIANCE is greater than or equal to Q3
q3_df = df[df['IRRADIANCE'] >= q3_irr]

# Define the temperature ranges

temp_ranges = [(20, 23), (23, 26), (26, 29), (29, 32), (32, 40)]


# Initialize a list to store the results
results = []

# Calculate the count and average power for each temperature range
for lower, upper in temp_ranges:
    # Filter the dataframe for the current temperature range
    temp_range_df = q3_df[(q3_df['TEMP PANEL'] >= lower) & (q3_df['TEMP PANEL'] < upper)]
    # Calculate the count and average power
    count = temp_range_df.shape[0]
    avg_power = temp_range_df['POWER'].mean()
    avg_irr = temp_range_df['IRRADIANCE'].mean()
    avg_eff = temp_range_df['EFFICIENCY'].mean()
    # Append the results
    results.append({'Temp Range': f'{lower} - {upper}', 'Count': count, 'Average Power': avg_power,
                    'Average Irradiance': avg_irr, 'Average Efficiency': avg_eff})

# Convert the results to a dataframe
results_df = pd.DataFrame(results)

# Display the dataframe
print(results_df)


# In[184]:


irradiance_min = df_irr['IRRADIANCE'].min()
irradiance_max = df_irr['IRRADIANCE'].max()

irradiance_range = np.linspace(irradiance_min, irradiance_max, 5)

# Initialize a list to store the results
results = []

# Calculate the count and average power for each irradiance range
for i in range(len(irradiance_range)-1):
    # Filter the dataframe for the current irradiance range
    range_df = df[(df['IRRADIANCE'] >= irradiance_range[i]) & (df['IRRADIANCE'] < irradiance_range[i+1])]
    # Calculate the count and average power
    count = range_df.shape[0]
    avg_power = range_df['POWER'].mean()
    avg_temp = range_df['TEMP PANEL'].mean()
    avg_eff = range_df['EFFICIENCY'].mean()
    #avg_tm = range_df['TIME'].mean()
    # Append the results
    results.append({'Irradiance Range': f'{irradiance_range[i]:.2f} - {irradiance_range[i+1]:.2f}', 'Count': count,
                    'Avg Power': avg_power,'Avg Temperature': avg_temp, 'Avg Efficiency': avg_eff})

# Convert the results to a dataframe
results_df = pd.DataFrame(results)

print(m_irr)
# Display the dataframe
print(results_df)


# In[185]:


irradiance_ranges = irradiance_range
# Initialize a list to store the results
results = []

# Iterate through each irradiance range
for i in range(len(irradiance_ranges)-1):
    # Filter the dataframe for the current irradiance range
    range_df = df[(df['IRRADIANCE'] >= irradiance_ranges[i]) & (df['IRRADIANCE'] < irradiance_ranges[i+1])]
    # Get the min and max for TEMP PANEL, EFFICIENCY, and POWER within the current range
    temp_panel_range = (round(range_df['TEMP PANEL'].min(), 2), round(range_df['TEMP PANEL'].max(), 2))
    efficiency_range = (round(range_df['EFFICIENCY'].min(), 2), round(range_df['EFFICIENCY'].max(), 2))
    power_range = (round(range_df['POWER'].min(), 2), round(range_df['POWER'].max(), 2))
    # Append the results
    results.append({'Irradiance Range': f'{irradiance_ranges[i]:.2f} - {irradiance_ranges[i+1]:.2f}',
                    'Temp Panel Range': temp_panel_range,
                    'Efficiency Range': efficiency_range,
                    'Power Range': power_range})


# Convert the results to a dataframe
results_df = pd.DataFrame(results)

# Display the dataframe
print(results_df)


# In[189]:


# Hacer indice al tiempo
df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' +
                                df['TIME'].astype(str))
df.set_index('DATETIME', inplace=True)

sorted_df = df.sort_values(by='IRRADIANCE', ascending=False)

# Tiempos donde la potencia es el mejor 10%
top_times = sorted_df[sorted_df['IRRADIANCE'] > irradiance_min].index.time
top_times = sorted_df[sorted_df['IRRADIANCE'] < irradiance_max].index.time

# Tiempos comunes mostrados de menor a mayor y por coincidencias

print(irradiance_min, irradiance_max)
print(top_times.min(),top_times.max())


# In[305]:


sheet1_df = pd.read_excel('DatosVFlotante.xlsx', sheet_name='Sheet1')
mdt_df = pd.read_excel('DatosVFlotante.xlsx', sheet_name='MDT')

# Merge the two dataframes to find the differences
merged_df = pd.merge(sheet1_df, mdt_df, how='outer', indicator=True)

# Filter out the rows that are only in the original dataset
removed_values_df = merged_df[merged_df['_merge'] == 'left_only']

# Plotting
fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(sheet1_df['Piranometro'], sheet1_df['Voltaje Flotante'], alpha=0.5, label='Datos registrados')
ax.scatter(removed_values_df['Piranometro'], removed_values_df['Voltaje Flotante'], color='red', alpha=0.5, label='Datos atípicos')
ax.set_xlabel('Irradiancia')
ax.set_ylabel('Voltaje')
ax.set_title('Comparación entre Datos originales y datos atípicos')
ax.legend()
plt.legend(loc=4)


fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(sheet1_df['Piranometro'], sheet1_df['Corriente Flotante'], alpha=0.5, label='Datos registrados')
ax.scatter(removed_values_df['Piranometro'], removed_values_df['Corriente Flotante'], color='red', alpha=0.5, label='Datos atípicos')
ax.set_xlabel('Irradiancia')
ax.set_ylabel('Corriente')
ax.legend()
plt.legend(loc=4)
plt.show()


# In[360]:


# Load the data
df_temp = pd.read_excel('temperaturas.xlsx')

# Create a boxplot for TEMP F and TEMP C
plt.figure(figsize=(5, 4))
df_temp.boxplot(column=['TEMP F', 'TEMP C'])
plt.title('Temperatura flotante y Temperatura convencional')
plt.ylabel('TEMPERATURE °C')
plt.grid(False)
plt.show()

