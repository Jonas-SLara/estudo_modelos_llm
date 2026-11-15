import os
import sys

# Tarefas armazenadas em um dicionário
tarefas = []

# Função para ordenar tarefas decrescentemente por prioridade
def ordenar_tarefas(prioridade):
    return sorted(tarefas, key=lambda t: (-t['prioridade'], -t['id']))

# Função para adicionar tarefas
def adicionar_tarefa(nome, descricao, id=None):
    if id is None:
        id = len(tarefas) + 1
    tarefas.append({
        'id': id,
        'nome': nome,
        'descricao': descricao,
        'pending': False
    })

# Função para listar tarefas
def listar_tarefas():
    if not tarefas:
        print("Nenhuma tarefa adicionada.")
        return
    print("Tarefas:")
    for tarefa in ordenar_tarefas(0):
        print(f"ID: {tarefa['id']}, Nome: {tarefa['nome']}, Descrição: {tarefa['descricao']}")
        print("Prioridade: ", tarefa['pending'] and "Sim" or "Não")

# Função para concluir tarefas
def concluir_tarefa(id):
    for tarefa in tarefas:
        if tarefa['id'] == id:
            tarefa['pending'] = True
            break

# Função para ordenar tarefas decrescentemente por prioridade
def ordenar_tarefas_decrescentemente():
    return sorted(tarefas, key=lambda t: (-t['prioridade'], -t['id']))

# Funções principais
def main():
    while True:
        print("\n1. Adicionar tarefa")
        print("2. Listar tarefas")
        print("3. Concluir tarefa")
        print("4. Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            nome = input("Nome: ")
            descricao = input("Descrição: ")
            id = None
            if id is None:
                id = len(tarefas) + 1
            adicionar_tarefa(nome, descricao, id)
        elif opcao == "2":
            listar_tarefas()
        elif opcao == "3":
            id_tarefa = int(input("ID da tarefa a concluir: "))
            concluir_tarefa(id_tarefa)
        elif opcao == "4":
            sys.exit(0)

if __name__ == "__main__":
    main()
