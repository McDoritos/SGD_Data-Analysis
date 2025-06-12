import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

def read_csv(file_name):
    df = pd.read_csv(file_name)
    print(f"File {file_name} read with success")
    return df

def read_xlsx(file_name, spreadsheet):
    if spreadsheet is None:
        df = pd.read_excel(file_name)
    else:
        df = pd.read_excel(file_name, sheet_name=spreadsheet)
            
    print(f"File {file_name} read with success")
    return df

def create_dataset_countries(df_fda,df_fde,df_gdpg,df_gdpe,df_bk,df_mac,df_obea,df_obee,df_paa,df_pae,df_fc,df_pope):
    # -------------------------------
    # 1. Seleção de nutrientes para o dataset americano
    # -------------------------------
    df_fda_treated = df_fda[
        (df_fda['Survey Years'] == '2017-2018') &
        (df_fda['Age Group'] == '2 and over') &
        (df_fda['Sex'] == 'All') &
        (df_fda['Race and Hispanic Origin'] == 'All')
    ]
    df_fda_treated = df_fda_treated[['Sex','Age Group','Nutrient', 'Mean']]

    print("Dados americanos de food diet tratados:")
    print(df_fda_treated)

    # -------------------------------
    # 2. Seleção de surveys europeus relevantes
    # -------------------------------
    selected_surveys = [
        'Austrian Study on Nutritional Status 2010-12 - Adults',
        'Diet National 2004',
        'National Survey of Food Intake and Nutrition',
        'Childhealth',
        'Czech National Food Consumption Survey',
        'The Danish National Dietary survey 2005-2008',
        'National Dietary Survey 1997',
        'National FINDIET 2012 Survey',
        'Individual and national study on food consumption 2',
        'National Nutrition Survey II',
        'Diet Lactation GR',
        'National Repr Surv',
        'National Adult Nutrition Survey',
        'Italian National Food Consumption Survey INRAN-SCAI 2005-06',
        'National Dietary Survey',
        'Dutch National food consumption survey 2007 - 2010',
        'National Food and Nutrition Institute - FAO 2000',
        'Dieta Pilot Adults',
        'SK MON 2008',
        'CRP-2008',
        'Spanish Agency for Food Safety (AESAN) Survey',
        'Swedish National Dietary Survey - Riksmaten adults 2010-11',
        'National Diet and Nutrition Survey - Years 1-3'
    ]
    df_selected_surveys = df_fde[df_fde['Survey'].isin(selected_surveys)]

    # -------------------------------
    # 3. Filtragem e agregação dos dados europeus por país e Foodex L1
    # -------------------------------
    df_filtered = df_selected_surveys[['Country', 'Pop Class', 'Foodex L1', 'Mean']]

    # Agrupar para calcular média por país e Foodex L1
    df_all_class = (
        df_filtered
        .groupby(['Country', 'Foodex L1'], as_index=False)
        .agg({'Mean': 'mean'})
    )

    df_all_class['Pop Class'] = 'All'

    df_all_class = df_all_class[['Country', 'Pop Class', 'Foodex L1', 'Mean']]

    df_fde_treated = pd.concat([df_filtered, df_all_class], ignore_index=True)

    df_fde_treated = df_fde_treated[df_fde_treated['Pop Class'] == 'All']

    print("Dados europeus de food diet tratados:")
    print(df_fde_treated)

    # -------------------------------
    # 5. Filtragem e tratamentos de datasets com GDP growth
    # -------------------------------
    print(df_gdpe.columns)
    unique_values = {
        'Country': df_gdpe['Country Name'].unique()
    }
#
    for col, values in unique_values.items():
        print(f"\n{col}:")
        for value in values:
            print(f" - {value}")

    df_gdpe = df_gdpe[['Country Name', 2017]]

    df_gdpg = df_gdpg[
        (df_gdpg['Country Name'] == 'United States')
    ]
    df_gdpg = df_gdpg[['Country Name', '2017']]
    print("Dados globais de GDP tratados:")
    print(df_gdpe)
    print(df_gdpg)

    # -------------------------------
    # 6. Tratamento de datasets de restaurantes Fast food
    # -------------------------------
    print("Dados globais de restaurantes FF tratados:")
    print(df_bk)
    df_mac = df_mac[['Country','Number of MacDonald restaurants']]
    df_mac['Number of MacDonald restaurants'] = (
        df_mac['Number of MacDonald restaurants']
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace('nan', np.nan)
        .astype(float)
    )
    print(df_mac)

    common_countries = set(df_bk['Country']).intersection(set(df_mac['Country']))
    common_countries = sorted(list(common_countries))

    df_common_ff_restaurants = pd.DataFrame({'Country': common_countries})

    bk_counts = df_bk.set_index('Country')['Number of Burger King Restaurants']
    df_common_ff_restaurants['BurgerKing_Count'] = df_common_ff_restaurants['Country'].map(bk_counts)

    mac_counts = df_mac.set_index('Country')['Number of MacDonald restaurants']
    df_common_ff_restaurants['McDonalds_Count'] = df_common_ff_restaurants['Country'].map(mac_counts)

    print("Datasets de restaurantes FF merged:")
    print(df_common_ff_restaurants)

    # -------------------------------
    # 7. Tratamento de datasets de obesidade
    # -------------------------------
    df_obee = df_obee[['Country','Overweight']]
    # Remover linhas com caracter  ":" na coluna 'Overweight'
    df_obee = df_obee[~df_obee['Overweight'].astype(str).str.contains(':')]
    print(df_obee)
    df_obea = df_obea[
        (df_obea['Survey Years'] == '2017-2018') &
        (df_obea['Age Group'] == '20 and over') &
        (df_obea['Sex'] == 'All') &
        (df_obea['Measure'] == 'Obesity') &
        (df_obea['Race and Hispanic Origin'] == 'All')
    ]
    df_obea = df_obea[['Sex','Age Group','Measure','Percent']]
    print(df_obea)

    # -------------------------------
    # 8. Tratamento de datasets de Atividade fisica
    # -------------------------------
    df_pae = df_pae[['Country','At least 4 times']]
    

    # Remove valores com ":"
    df_pae = df_pae[~df_pae['At least 4 times'].astype(str).str.contains(':')]
    print(df_pae)
    df_paa = df_paa[
        (df_paa['YearStart'] == 2017)
    ]
    df_paa = df_paa[['YearStart','Class','LocationDesc','Data_Value']]
    print(df_paa)

    # -------------------------------
    # 9. Merge dos dados tratados num só dataset para a europa
    # -------------------------------
    all_countries = set()
    all_countries.update(df_fde_treated['Country'].unique())
    all_countries.update(df_gdpe['Country Name'].unique())
    all_countries.update(df_common_ff_restaurants['Country'].unique())
    all_countries.update(df_obee['Country'].unique())
    all_countries.update(df_pae['Country'].unique())

    df_final = pd.DataFrame({'Country': sorted(list(all_countries))})

    # 2. Transposição de Categorias alimentares para colunas
    food_consumption_pivoted = df_fde_treated.pivot_table(
        index='Country',
        columns='Foodex L1',
        values='Mean',
        aggfunc='mean'
    ).reset_index()

    # Limpar nomes das colunas
    food_consumption_pivoted.columns = [
        f"Consumo_{col.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}" 
        if col != 'Country' else col 
        for col in food_consumption_pivoted.columns
    ]

    # Identificar colunas de consumo
    consumo_cols = [col for col in food_consumption_pivoted.columns if col != 'Country']
    food_consumption_pivoted[consumo_cols] = food_consumption_pivoted[consumo_cols].round(2)

    # Juntar ao DataFrame final
    df_final = pd.merge(df_final[['Country']],
                    food_consumption_pivoted,
                    on='Country',
                    how='left')

    # 4. Juntar dados de gdp
    df_final = pd.merge(df_final, 
                    df_gdpe.rename(columns={'Country Name': 'Country', 2017: 'GDP_2017'}),
                    on='Country',
                    how='left')

    # 5. Juntar dados de restaurantes fast-food
    df_final = pd.merge(df_final,
                    df_common_ff_restaurants,
                    on='Country',
                    how='left')

    # 6. Juntar dados de obesidade (Europa)
    df_final = pd.merge(df_final,
                    df_obee.rename(columns={'Overweight': 'Obesity_Rate_Europe'}),
                    on='Country',
                    how='left')

    # 7. Juntar dados de atividade física
    df_final = pd.merge(df_final,
                    df_pae.rename(columns={'At least 4 times': 'Physical_Activity_4plus_times'}),
                    on='Country',
                    how='left')

    # 8. Adicionar referência de obesidade dos EUA
    us_obesity = df_obea['Percent'].values[0] if len(df_obea) > 0 else None
    df_final['US_Obesity_Reference'] = us_obesity

    # 9. Adição dos dados populacionais
    df_pope_clean = df_pope[['Country', 'Population Density', 'Population', 'Area']].copy()

    # Padronizar nomes de países
    df_pope_clean['Country'] = df_pope_clean['Country'].str.strip().str.title()
    df_final['Country'] = df_final['Country'].str.strip().str.title()

    df_final = pd.merge(df_final,
                    df_pope_clean,
                    on='Country',
                    how='left')

    # Renomeação
    df_final = df_final.rename(columns={
        'Population Density': 'Pop_Density',
        'Area': 'Area_km2'
    })

    # Reordenação de colunas
    first_cols = ['Country', 'Population', 'Pop_Density', 'Area_km2']
    other_cols = [col for col in df_final.columns if col not in first_cols]
    df_final = df_final[first_cols + other_cols]

    # Remoção e tratamento de dados e colunas
    df_final = df_final[df_final['Obesity_Rate_Europe'].notna()]
    df_final = df_final[df_final['Consumo_Alcoholic_beverages'].notna()]
    df_final = df_final.drop(columns=['Consumo_Food_for_infants_and_small_children'])
    df_final = df_final.drop(columns=['Consumo_Products_for_special_nutritional_use'])
    df_final['Physical_Activity_4plus_times'] = df_final['Physical_Activity_4plus_times'].fillna(
    round(df_final['Physical_Activity_4plus_times'].mean(),2))

    removed_countries = set(all_countries) - set(df_final['Country'])
    if removed_countries:
        print("\nPaíses removidos por falta de dados de obesidade:")
        print(sorted(removed_countries))

    df_final = df_final.sort_values('Country').reset_index(drop=True)

    print("Dataset consolidado final com todos os dados populacionais:")
    print(df_final.head())
    df_final.to_csv('consolidated_data_by_country.csv', index=False)

    return df_final

def create_features(df_final):
    cols_to_convert = ['BurgerKing_Count', 'McDonalds_Count', 'Population', 'Area_km2']

    for col in cols_to_convert:
        df_final[col] = pd.to_numeric(df_final[col].astype(str).str.replace(',', '.'), errors='coerce')


    if all(col in df_final.columns for col in cols_to_convert):
        missing_values = df_final[cols_to_convert].isnull().sum()
        if missing_values.any():
            print("\nValores faltantes após conversão:")
            print(missing_values)

        df_final['Total_FastFood_Count'] = df_final['BurgerKing_Count'].fillna(0) + df_final['McDonalds_Count'].fillna(0)

        with np.errstate(divide='ignore', invalid='ignore'):
            df_final['FastFood_per_100k'] = np.where(
                df_final['Population'] > 0,
                (df_final['Total_FastFood_Count'] / df_final['Population']) * 100000,
                np.nan
            )

            df_final['FastFood_density_per_1000km2'] = np.where(
                df_final['Area_km2'] > 0,
                (df_final['Total_FastFood_Count'] / df_final['Area_km2']) * 1000,
                np.nan
            )

        round_cols = ['FastFood_per_100k', 'FastFood_density_per_1000km2']
        df_final[round_cols] = df_final[round_cols].round(2)

        ff_cols = ['Country', 'Population', 'Area_km2',
                'BurgerKing_Count', 'McDonalds_Count', 'Total_FastFood_Count'] + round_cols
        other_cols = [col for col in df_final.columns if col not in ff_cols]
        df_final = df_final[ff_cols + other_cols]

    else:
        print("Não foi possível calcular as métricas")

    df_final.to_csv('consolidated_data_by_country.csv', index=False)
    return df_final

def mainMenu():
    print(" Main Menu\n"
          "1 - Create dataset\n"
          "2 - Data analysis\n" 
          "3 - Linear Regression\n" 
          "4 - Ridge Regression\n" 
          "5 - Random Forest Regressor\n"
          "0 - Exit program\n")
    
def executeOption(option):
    if option == '1':
        foodDiet_america = read_csv("Datasets/Treated/Food-diet_America.csv")
        foodDiet_europe = read_xlsx("Datasets/Treated/Food-diet_Europe.xlsx","L1_Consuming_days_only_g_day_bw")
        foodComposition = read_csv("Datasets/Treated/Food-Composition_FoodexL1.csv")

        gdp_global = read_csv("Datasets/Treated/GDP-Growth_Countries.csv")
        gdp_europe = read_xlsx("Datasets/Treated/GDP-Growth_Europe.xlsx",'Folha1')

        bk_global = read_csv("Datasets/Treated/Num-BKs_Countries.csv")
        mac_global = read_csv("Datasets/Treated/Num-MACs_Countries.csv")

        obesity_america = read_csv("Datasets/Treated/Obesity_America.csv")
        obesity_europe = read_xlsx("Datasets/Treated/Obesity_Europe.xlsx","Folha1")

        physicalActivity_america = read_xlsx("Datasets/Treated/Physical-Activity_America.xlsx","Nutrition__Physical_Activity__a")
        physicalActivity_europe = read_xlsx("Datasets/Treated/Physical-Activity_Europe.xlsx","Folha1")

        Population_europe = read_csv("Datasets/Treated/Population_Europe.csv")
        df_treated = create_dataset_countries(foodDiet_america,foodDiet_europe,gdp_global,gdp_europe,bk_global,mac_global,obesity_america,obesity_europe,physicalActivity_america,physicalActivity_europe,foodComposition,Population_europe)
        df_final = create_features(df_treated)
        print(df_final)
    elif option == '2':
        df_treated = read_csv("consolidated_data_by_country.csv")
        print(df_treated.columns)
        # Matriz de Correlação
        correlation_matrix = df_treated.corr(numeric_only=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Matriz de Correlação")
        plt.show()

        # Distribuição da Taxa de Obesidade
        plt.figure(figsize=(10, 6))
        sns.histplot(df_treated['Obesity_Rate_Europe'], bins=15, kde=True)
        plt.title("Distribuição da Taxa de Obesidade")
        plt.xlabel("Taxa de Obesidade")
        plt.ylabel("Frequência")
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Gráfico Legumes, Nozes e Sementes
        sns.regplot(
            data=df_treated,
            x='Consumo_Legumes_nuts_and_oilseeds',
            y='Obesity_Rate_Europe',
            scatter_kws={'s': 70},
            line_kws={'color': 'green'},
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("Legumes, Nozes e Sementes vs. Obesidade")
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("Taxa de Obesidade")

        # Gráfico Água Potável
        sns.regplot(
            data=df_treated,
            x='Consumo_Drinking_water_water_without_any_additives_except',
            y='Obesity_Rate_Europe',
            scatter_kws={'s': 70},
            line_kws={'color': 'blue'},
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Água Potável vs. Obesidade")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Taxa de Obesidade")

        # Gráfico Leite e Derivados
        sns.regplot(
            data=df_treated,
            x='Consumo_Milk_and_dairy_products',
            y='Obesity_Rate_Europe',
            scatter_kws={'s': 70},
            line_kws={'color': 'purple'},
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Leite e Derivados vs. Obesidade")
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Taxa de Obesidade")

        # Gráfico Alimentos Compostos / Congelados
        sns.regplot(
            data=df_treated,
            x='Consumo_Composite_food_including_frozen_products',
            y='Obesity_Rate_Europe',
            scatter_kws={'s': 70},
            line_kws={'color': 'orange'},
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("Alimentos Compostos/Congelados vs. Obesidade")
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Taxa de Obesidade")

        plt.tight_layout()
        plt.show()

        # Gráfico Densidade Populacional vs. Taxa de Obesidade
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df_treated, x='Pop_Density', y='Obesity_Rate_Europe', scatter_kws={'s': 100}, line_kws={'color': 'red'})
        plt.title("Relação entre Densidade Populacional e Taxa de Obesidade")
        plt.xlabel("Densidade Populacional")
        plt.ylabel("Taxa de Obesidade")
        plt.show()

        # Gráfico PIB vs. Consumo de Bebidas Alcoólicas
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df_treated, x='GDP_2017', y='Consumo_Alcoholic_beverages', scatter_kws={'s': 100}, line_kws={'color': 'red'})
        plt.title("Relação entre PIB e Consumo de Bebidas Alcoólicas")
        plt.xlabel("PIB")
        plt.ylabel("Consumo de Bebidas Alcoólicas")
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

        # Gráfico Atividade Física vs. Consumo de Bebidas Alcoólicas
        sns.regplot(
            data=df_treated,
            x='Physical_Activity_4plus_times',
            y='Consumo_Alcoholic_beverages',
            scatter_kws={'s': 60},
            line_kws={'color': 'blue'},
            ax=axes[0]
        )
        axes[0].set_title("Atividade Física vs. Bebidas Alcoólicas")
        axes[0].set_xlabel("Atividade Física (4+ vezes/semana)")
        axes[0].set_ylabel("Consumo de Bebidas Alcoólicas")

        # Gráfico Atividade Física vs. Consumo de Bebidas Não Alcoólicas
        sns.regplot(
            data=df_treated,
            x='Physical_Activity_4plus_times',
            y='Consumo_Non-alcoholic_beverages_excepting_milk_based_beve',
            scatter_kws={'s': 60},
            line_kws={'color': 'red'},
            ax=axes[1]
        )
        axes[1].set_title("Atividade Física vs. Bebidas Não Alcoólicas")
        axes[1].set_xlabel("Atividade Física (4+ vezes/semana)")
        axes[1].set_ylabel("Consumo de Bebidas Não Alcoólicas")

        # Gráfico Atividade Física vs. Consumo de Peixes e Frutos do Mar
        sns.regplot(
            data=df_treated,
            x='Physical_Activity_4plus_times',
            y='Consumo_Fish_and_other_seafood_including_amphibians_rept',
            scatter_kws={'s': 60},
            line_kws={'color': 'green'},
            ax=axes[2]
        )
        axes[2].set_title("Atividade Física vs. Consumo de Peixes e Frutos do Mar")
        axes[2].set_xlabel("Atividade Física (4+ vezes/semana)")
        axes[2].set_ylabel("Consumo de Peixes e Frutos do Mar")

        plt.tight_layout()
        plt.show()

        # Gráfico Densidade Populacional vs. Densidade de Fast Food
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df_treated, x='Pop_Density', y='FastFood_density_per_1000km2', scatter_kws={'s': 100}, line_kws={'color': 'red'})
        plt.title("Relação entre Densidade Populacional e Densidade de Fast Food")
        plt.xlabel("Densidade Populacional")
        plt.ylabel("Densidade de Fast Food por 1000 km²")
        plt.show()

    elif option == '3':
        df = pd.read_csv("consolidated_data_by_country.csv")

        x = df.drop(columns=['Country', 'US_Obesity_Reference', 'Obesity_Rate_Europe'])
        y = df['Obesity_Rate_Europe']

        print("All features")
        calcRegLinear(x, y)
    elif option == '4':
        df = pd.read_csv("consolidated_data_by_country.csv")

        x = df.drop(columns=['Country', 'US_Obesity_Reference', 'Obesity_Rate_Europe'])
        y = df['Obesity_Rate_Europe']

        print("All features")
        calcRidge(x, y)
    elif option == '5':
        df = pd.read_csv("consolidated_data_by_country.csv")

        x = df.drop(columns=['Country', 'US_Obesity_Reference', 'Obesity_Rate_Europe'])
        y = df['Obesity_Rate_Europe']

        print("All features")
        calcRandomForestRegressor(x, y)
    elif option == '0':
        print("Exiting program...")
    else:
        print("Invalid option")

def calcRegLinear(x,y):
    scaler = StandardScaler()

    X_normalized = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.4, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2 score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))

    resultados = pd.DataFrame({
        'Real': y_test.values,
        'Previsto': y_pred,
        'Erro (Previsto - Real)': y_pred - y_test.values,
        'Erro Absoluto': abs(y_pred - y_test.values)
    })

    resultados = resultados.reset_index(drop=True)

    print(resultados)

def calcRidge(x,y):
    scaler = StandardScaler()

    X_normalized = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.4, random_state=42)

    model = Ridge(alpha=1.0) 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2 score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))

    resultados = pd.DataFrame({
        'Real': y_test.values,
        'Previsto': y_pred,
        'Erro (Previsto - Real)': y_pred - y_test.values,
        'Erro Absoluto': abs(y_pred - y_test.values)
    })

    resultados = resultados.reset_index(drop=True)

    print(resultados)

def calcRandomForestRegressor(x,y):
    scaler = StandardScaler()

    X_normalized = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.4, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2 score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))

    resultados = pd.DataFrame({
        'Real': y_test.values,
        'Previsto': y_pred,
        'Erro (Previsto - Real)': y_pred - y_test.values,
        'Erro Absoluto': abs(y_pred - y_test.values)
    })

    resultados = resultados.reset_index(drop=True)

    print(resultados)

option = -1
while(option!='0'):
    mainMenu()
    option = input("Choose an option: ")
    executeOption(option)