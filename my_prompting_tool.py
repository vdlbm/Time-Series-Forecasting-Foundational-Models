import os
import openai
import pandas as pd

import re

#from data1.serialize import SerializerSettings

# Configuración de la API de OpenAI
openai.api_key = ''  # Clave de API de OpenAI (debe ser proporcionada)
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")  # URL base de la API

# Selección del modelo a utilizar
# model_select = 'gpt-4-1106-preview'  # Opción alternativa
model_select = 'llama2-13b-chat'  # Modelo seleccionado

# Hiperparámetros para GPT-3.5
'''

gpt3_hypers = dict(
    temp=0.7,  # Temperatura para el muestreo
    alpha=0.95,  # Parámetro de escalado
    beta=0.3,  # Parámetro de desplazamiento
    basic=False,  # Si se usa la versión básica del escalado
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)  # Configuración de serialización
)

# Hiperparámetros para GPT-4
gpt4_hypers = dict(
    alpha=0.3,  # Parámetro de escalado
    basic=True,  # Si se usa la versión básica del escalado
    temp=0.5,  # Temperatura para el muestreo
    top_p=0.5,  # Parámetro de muestreo top-p
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')  # Configuración de serialización
)

# Mensajes de hiperparámetros para GPT-3.5 y GPT-4 (en inglés, ya que son parte del prompt)
hyper_gpt35 = f"Set hyperparameters to: temp={gpt3_hypers['temp']}, " \
              f"alpha={gpt3_hypers['alpha']}, beta={gpt3_hypers['beta']}, " \
              f"basic={gpt3_hypers['basic']}, settings={gpt3_hypers['settings']}"

hyper_gpt4 = f"Set hyperparameters to: alpha={gpt4_hypers['alpha']}, " \
             f"temp={gpt4_hypers['temp']}, top_p={gpt4_hypers['top_p']}, " \
             f"basic={gpt4_hypers['basic']}, settings={gpt4_hypers['settings']}"

'''

# Función para generar una descripción inicial de cada conjunto de datos (en inglés, ya que es parte del prompt)
def paraphrase_initial(data_name):
    
    if data_name == 'ETH_MONTHS':
        desp = "This is a monthly time series dataset describing the price of Ethereum (ETH) in US dollars (USD), " \
               "each value represents the average price of Ethereum in USD for that month. "
    elif data_name == 'ETH_DAYS':
        desp = "This is a daily time series dataset describing the price of Ethereum (ETH) in US dollars (USD), " \
               "each value represents the average price of Ethereum in USD for that day. "
    elif data_name == 'ETH_HOURS':
        desp = "This is an hourly time series dataset describing the price of Ethereum (ETH) in US dollars (USD), " \
               "each value represents the average price of Ethereum in USD for that hour. "
    else:
        desp = "Description not available for this dataset. "

    return desp

# Función para convertir una secuencia en lenguaje natural 
def paraphrase_seq2lan(seq, desp):
    results = ''
    # Lee los valores de la secuencia uno por uno y genera una descripción
    for i in range(len(seq) - 1):
        t1 = seq.iloc[i]  # Selecciona el elemento actual
        t2 = seq.iloc[i + 1]  # Selecciona el siguiente elemento
        result = describe_change(t1, t2)  # Describe el cambio entre t1 y t2
        results += result
    lan = desp + results  # Combina la descripción inicial con los cambios

    return lan


# Función para describir el cambio entre dos valores 
def describe_change(t1, t2):
    try:
        t11 = t1.item()
        t22 = t2.item()
    except Exception as e:
        print("Error en describe_change:", t1, t2, e)
        return ""  # devuelve vacío para no romper todo
    
    if pd.isna(t11) or pd.isna(t22):
        print("NaN detectado en:", t1, t2)
        return ""  # evita generar frase
    
    if t22 > t11:
        return f"from {t11} increasing to {t22}, "
    elif t22 < t11:
        return f"from {t11} decreasing to {t22}, "
    else:
        return f"it remains flat from {t11} to {t22}, "


# Función para recuperar una secuencia a partir de una descripción en lenguaje natural
def recover_lan2seq(input_string):
    # Paso 1: Elimina la descripción inicial
    dot_index = input_string.find('.')
    cleaned_string = input_string[dot_index + 1:].strip() if dot_index != -1 else input_string.strip()

    # Paso 2: Extrae los números de la cadena
    numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned_string)
    # Convierte los números a tipo float
    float_numbers = [float(num) for num in numbers]

    # Paso 3: Elimina los números duplicados
    recovered = [float_numbers[0]]
    for i in range(1, len(float_numbers), 2):  # tomar cada "segundo" del par
        recovered.append(float_numbers[i])
    # Convierte la lista en un DataFrame de pandas
    result_series = pd.DataFrame(recovered)

    return result_series



# Función similar a recover_lan2seq pero específica para LLM
def recover_lan2seq_llm(input_string):
    # Extrae los números de la cadena
    numbers = re.findall(r'(\d+\.\d+)', input_string)
    # Convierte los números a tipo float
    float_numbers = [float(num) for num in numbers]

    # Elimina los números duplicados
    filtered_numbers = [float_numbers[i] for i in range(len(float_numbers)) if i % 2 == 0]
    # Añade el último número
    filtered_numbers.append(float_numbers[-1])
    # Convierte la lista en una Serie de pandas
    result_series = pd.Series(filtered_numbers)

    return result_series

# Función para procesar un conjunto de datos y convertirlo en lenguaje natural
def paraphrase_nlp(dataset_name, train, test):
    desp = paraphrase_initial(dataset_name)  # Obtiene la descripción inicial

    print("Longitud del entrenamiento:", train.shape)
    print("Longitud de la prueba:", test.shape)

    # Convertir a lenguaje natural
    Train_lan = paraphrase_seq2lan(train, desp)  
    Test_lan = paraphrase_seq2lan(test, desp)  

    # Recuperar la secuencia de prueba desde el lenguaje natural
    seq_test = recover_lan2seq(Test_lan)  

    print("Longitud de la secuencia predicha:", seq_test.shape)
    
    if test.shape != seq_test.shape:
        print("Warning! Se perdieron datos.")
        print(seq_test.head())


    return Train_lan, Test_lan, seq_test


# Función para procesar un conjunto de datos utilizando un modelo de lenguaje (LLM)
def paraphrase_llm(datasets_list):
    prompt = "Analyze this time series and rewrite it " \
             "as a trend-by-trend representation of discrete values. Only numerical " \
             "changes are described, not date changes. For example, the template like {from 1.0 increasing to 2.0, " \
             "from 2.0 decreasing to 0.5,}. Be careful not to lose every sequence value. "
    

    for dataset_name in datasets_list:
        desp = paraphrase_initial(dataset_name)  # Obtiene la descripción inicial
        data = datasets[dataset_name]
        train, test = data
        content_train = "You are a useful assistant," + desp + str(train)  # Prepara el contenido para el entrenamiento
        content_test = "You are a useful assistant," + desp + str(test)  # Prepara el contenido para la prueba
        response = openai.ChatCompletion.create(
            model=model_select,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content_train}
            ]
        )
        Train_lan = response.choices[0].message.content  # Obtiene la respuesta del modelo
        response = openai.ChatCompletion.create(
            model=model_select,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content_test}
            ]
        )
        Test_lan = response.choices[0].message.content  # Obtiene la respuesta del modelo
        print("Test_lan:", Test_lan)
        seq_test = recover_lan2seq_llm(Test_lan)  # Recupera la secuencia de prueba
        if test.shape != seq_test.shape:
            print("Error in the process!")
            print("Shape of seq_test:", seq_test.shape)
            print("Shape of test:", test.shape)

    return Train_lan, Test_lan, seq_test

# Función para realizar predicciones utilizando un modelo de lenguaje (LLM)
def paraphrasing_predict_llm(desp, train_lan, steps, model_name):
    '''
    if model_name == 'gpt-3.5-turbo-0125':
        hyper_parameters_message = hyper_gpt35  # Usa los hiperparámetros de GPT-3.5
    else:
        hyper_parameters_message = hyper_gpt4  # Usa los hiperparámetros de GPT-4
    '''
    prompt = "You are a helpful assistant that performs time series predictions. " \
             "The user will provide a sequence and you will predict the remaining sequence." \
             "The sequence is represented by decimal strings separated by commas. " \
             "Please continue the following sequence without producing any additional text. " \
             "Do not say anything like 'the next terms in the sequence are', just return the numbers."
    prompt_add = (f"Predict the next {steps} steps, where each step follows the format (from 1.0 increasing to 2.0) or (from 2.0 decreasing to 0.5)."\
                  " The final output should precisely follow the specified number of steps. Provide a sequence:\n")

    content_train = prompt + desp + prompt_add + train_lan  # Prepara el contenido para la predicción
    response = openai.ChatCompletion.create(
        model=model_name,
        response_format={"type": "text"},
        messages=[
           # {"role": "system", "content": hyper_parameters_message},
            {"role": "system", "content": prompt},
            {"role": "user", "content": content_train}
        ]
    )
    Test_lan = response.choices[0].message.content  # Obtiene la respuesta del modelo
    seq_test = recover_lan2seq_llm(Test_lan)  # Recupera la secuencia predicha

    return seq_test
'''
# Función para realizar predicciones utilizando el modelo LLaMA
def paraphrasing_predict_llama(desp, train_lan, steps, model_name):
    prompt_add = f"Predict the next {steps} steps, where each step follows the format (from 1.0 increasing to 2.0) or " \
                 "(from 2.0 decreasing to 0.5). The final output should precisely follow the specified number of steps. Sequence:\n"
    content_train = desp + prompt_add + train_lan  # Prepara el contenido para la predicción
    response = llama_api_qa(model_name, content_train)  # Obtiene la respuesta del modelo LLaMA
    seq_test = recover_lan2seq_llm(response)  # Recupera la secuencia predicha

    return seq_test
'''
# Función principal de prueba
if __name__ == '__main__':
    # Lista de conjuntos de datos a procesar
    datasets_list = [
        'AirPassengersDataset',
        'AusBeerDataset',
        'GasRateCO2Dataset',
        'MonthlyMilkDataset',
        'SunspotsDataset',
        'WineDataset',
        'WoolyDataset',
        'HeartRateDataset',
    ]

    # Procesamiento tradicional (sin LLM)
    paraphrase_nlp(datasets_list)

    # Procesamiento utilizando un modelo de lenguaje (LLM)
    paraphrase_llm(datasets_list)