from django.shortcuts import render
from django.http import JsonResponse
from .Copilot import Chatbot  # Importe o chatbot do arquivo que você criou

# Instancia o chatbot
chatbot_instance = Chatbot()

def copilot(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        if user_message.lower() == 'sair':
            return JsonResponse({"response": "Sessão encerrada. Volte sempre!"})
        
        # Processa a mensagem do usuário
        df_produtos = chatbot_instance.conectar_banco_dados()
        resposta_saudacao = chatbot_instance.lidar_com_saudacoes(user_message)

        if resposta_saudacao:
            response = resposta_saudacao
        else:
            resposta_maximo = chatbot_instance.Filtropreco.filtrar_produtos_por_valor_maximo(user_message)
            if resposta_maximo is not None and not resposta_maximo.empty:
                response = resposta_maximo.to_json(orient='records')
            else:
                resposta_minimo = chatbot_instance.Filtropreco.filtrar_produtos_por_valor_minimo(user_message)
                if resposta_minimo is not None and not resposta_minimo.empty:
                    response = resposta_minimo.to_json(orient='records')
                else:
                    resposta_intervalo = chatbot_instance.Filtropreco.filtrar_produtos_por_intervalo(user_message)
                    if resposta_intervalo is not None and not resposta_intervalo.empty:
                        response = resposta_intervalo.to_json(orient='records')
                    else:
                        response = chatbot_instance.fazer_pergunta(chatbot_instance.create_training_data(df_produtos), user_message, df_produtos)
        
        return JsonResponse({"response": response})

    return render(request, "Main.html")
