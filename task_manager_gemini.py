import sys

# Tarefas armazenadas em memória: lista de dicionários
tarefas = []
proximo_id = 1

PRIORIDADE_MIN = 0
PRIORIDADE_MAX = 5


def adicionar_tarefa(nome, descricao, prioridade=0):
    """Cria uma tarefa com ID auto incrementado. pending=True significa pendente."""
    global proximo_id
    tarefa = {
        "id": proximo_id,
        "nome": nome,
        "descricao": descricao,
        "pending": True,
        "prioridade": prioridade,
    }
    tarefas.append(tarefa)
    proximo_id += 1
    return tarefa


def ordenar_tarefas():
    """Maior prioridade primeiro; em empate, a mais antiga (menor ID) primeiro."""
    return sorted(tarefas, key=lambda t: (-t["prioridade"], t["id"]))


def listar_tarefas():
    if not tarefas:
        print("Nenhuma tarefa adicionada.")
        return
    print("Tarefas (por prioridade, da maior para a menor):")
    for t in ordenar_tarefas():
        status = "Pendente" if t["pending"] else "Concluída"
        print(
            f"ID: {t['id']} | Prioridade: {t['prioridade']} | "
            f"Nome: {t['nome']} | Descrição: {t['descricao']} | Status: {status}"
        )


def concluir_tarefa(id_tarefa):
    for t in tarefas:
        if t["id"] == id_tarefa:
            t["pending"] = False
            return True
    return False


def ler_prioridade():
    """Pergunta a prioridade até receber um valor válido. Enter = 0."""
    while True:
        texto = input(f"Prioridade ({PRIORIDADE_MIN} a {PRIORIDADE_MAX}, Enter para 0): ").strip()
        if texto == "":
            return 0
        try:
            valor = int(texto)
        except ValueError:
            print("Digite um número inteiro.")
            continue
        if PRIORIDADE_MIN <= valor <= PRIORIDADE_MAX:
            return valor
        print(f"A prioridade deve estar entre {PRIORIDADE_MIN} e {PRIORIDADE_MAX}.")


def main():
    while True:
        print("\n1. Adicionar tarefa")
        print("2. Listar tarefas")
        print("3. Concluir tarefa")
        print("4. Sair")
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            nome = input("Nome: ")
            descricao = input("Descrição: ")
            prioridade = ler_prioridade()
            tarefa = adicionar_tarefa(nome, descricao, prioridade)
            print(f"Tarefa {tarefa['id']} adicionada.")
        elif opcao == "2":
            listar_tarefas()
        elif opcao == "3":
            try:
                id_tarefa = int(input("ID da tarefa a concluir: "))
            except ValueError:
                print("Digite um número inteiro.")
                continue
            if concluir_tarefa(id_tarefa):
                print("Tarefa concluída.")
            else:
                print("Tarefa não encontrada.")
        elif opcao == "4":
            sys.exit(0)
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()