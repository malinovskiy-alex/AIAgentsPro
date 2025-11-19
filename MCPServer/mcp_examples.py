"""
Приклади використання різних MCP серверів з CrewAI

Демонструє інтеграцію з популярними офіційними MCP серверами:
- Sequential Thinking - структуроване мислення
- Filesystem - робота з файловою системою
- Fetch - завантаження веб-контенту
- Git - робота з Git репозиторіями
- Memory - knowledge graph пам'ять
"""

from crewai import Agent, Task, Crew
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from config import Config

# Initialize configuration
Config.validate()


# ============================================================================
# 1. SEQUENTIAL THINKING - Структуроване мислення
# ============================================================================

def example_sequential_thinking():
    """
    Приклад використання Sequential Thinking MCP сервера
    Використовується для розбиття складних проблем на логічні кроки
    """
    print("\n" + "="*80)
    print("📊 ПРИКЛАД 1: Sequential Thinking MCP Server")
    print("="*80 + "\n")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    )

    with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
        print(f"✅ Підключено до Sequential Thinking сервера")
        print(f"   Доступно інструментів: {len(mcp_tools)}\n")

        agent = Agent(
            role='Problem Solver',
            goal='Розв\'язувати складні проблеми використовуючи структуроване мислення',
            backstory='Ти експерт з розбиття складних завдань на прості логічні кроки.',
            tools=mcp_tools,
            verbose=True
        )

        task = Task(
            description='''
            Використай інструмент sequentialthinking для аналізу проблеми:
            "Як оптимізувати процес розробки ПЗ в команді з 5 розробників?"

            Виконай 3 кроки:
            1. Визначення проблеми (thoughtNumber: 1, totalThoughts: 3)
            2. Аналіз рішень (thoughtNumber: 2, totalThoughts: 3)
            3. Рекомендації (thoughtNumber: 3, totalThoughts: 3)
            ''',
            agent=agent,
            expected_output='Структурований аналіз з 3 кроків'
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        print(f"\n✅ Результат: {result}")


# ============================================================================
# 2. FILESYSTEM - Робота з файловою системою
# ============================================================================

def example_filesystem():
    """
    Приклад використання Filesystem MCP сервера
    Безпечна робота з файлами з контролем доступу
    """
    print("\n" + "="*80)
    print("📁 ПРИКЛАД 2: Filesystem MCP Server")
    print("="*80 + "\n")

    # ВАЖЛИВО: Вказати allowed_directories для безпеки!
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp",  # Дозволена директорія
            str(Config.__file__.replace('config.py', ''))  # Поточна директорія проекту
        ]
    )

    with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
        print(f"✅ Підключено до Filesystem сервера")
        print(f"   Доступно інструментів: {len(mcp_tools)}")

        if mcp_tools:
            print(f"   📋 Інструменти:")
            for tool in mcp_tools:
                print(f"      • {tool.name}")
        print()

        agent = Agent(
            role='File Manager',
            goal='Керувати файлами та директоріями',
            backstory='Ти експерт з організації файлової системи.',
            tools=mcp_tools,
            verbose=True
        )

        task = Task(
            description='''
            Використай доступні інструменти для:
            1. Перегляду списку файлів у поточній директорії
            2. Читання файлу README.md якщо він існує
            3. Створення звіту про знайдені .py файли
            ''',
            agent=agent,
            expected_output='Звіт про файли в директорії'
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        print(f"\n✅ Результат: {result}")


# ============================================================================
# 3. FETCH - Завантаження веб-контенту
# ============================================================================

def example_fetch():
    """
    Приклад використання Fetch MCP сервера
    Завантаження та конвертація веб-контенту для LLM
    """
    print("\n" + "="*80)
    print("🌐 ПРИКЛАД 3: Fetch MCP Server")
    print("="*80 + "\n")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-fetch"]
    )

    with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
        print(f"✅ Підключено до Fetch сервера")
        print(f"   Доступно інструментів: {len(mcp_tools)}")

        if mcp_tools:
            print(f"   📋 Інструменти:")
            for tool in mcp_tools:
                print(f"      • {tool.name}")
        print()

        agent = Agent(
            role='Web Content Researcher',
            goal='Досліджувати та аналізувати веб-контент',
            backstory='Ти експерт з пошуку та аналізу інформації в інтернеті.',
            tools=mcp_tools,
            verbose=True
        )

        task = Task(
            description='''
            Використай fetch інструмент для:
            1. Завантаження документації з https://docs.crewai.com
            2. Аналізу основних можливостей CrewAI
            3. Створення короткого резюме (2-3 речення)
            ''',
            agent=agent,
            expected_output='Короткий аналіз документації CrewAI'
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        print(f"\n✅ Результат: {result}")


# ============================================================================
# 4. GIT - Робота з Git репозиторіями
# ============================================================================

def example_git():
    """
    Приклад використання Git MCP сервера
    Читання, пошук та аналіз Git репозиторіїв
    """
    print("\n" + "="*80)
    print("🔧 ПРИКЛАД 4: Git MCP Server")
    print("="*80 + "\n")

    # Вказуємо поточну директорію як Git репозиторій
    import os
    repo_path = os.getcwd()

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-git", "--repository", repo_path]
    )

    with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
        print(f"✅ Підключено до Git сервера")
        print(f"   Репозиторій: {repo_path}")
        print(f"   Доступно інструментів: {len(mcp_tools)}")

        if mcp_tools:
            print(f"   📋 Інструменти:")
            for tool in mcp_tools:
                print(f"      • {tool.name}")
        print()

        agent = Agent(
            role='Git Repository Analyst',
            goal='Аналізувати Git репозиторії та надавати інсайти',
            backstory='Ти експерт з аналізу Git історії та структури коду.',
            tools=mcp_tools,
            verbose=True
        )

        task = Task(
            description='''
            Використай Git інструменти для:
            1. Отримання останніх 3 комітів
            2. Аналізу змін у проекті
            3. Створення короткого звіту про розробку
            ''',
            agent=agent,
            expected_output='Звіт про останні зміни в репозиторії'
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        print(f"\n✅ Результат: {result}")


# ============================================================================
# 5. MEMORY - Knowledge Graph пам'ять
# ============================================================================

def example_memory():
    """
    Приклад використання Memory MCP сервера
    Система пам'яті на основі knowledge graph
    """
    print("\n" + "="*80)
    print("🧠 ПРИКЛАД 5: Memory MCP Server")
    print("="*80 + "\n")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"]
    )

    with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
        print(f"✅ Підключено до Memory сервера")
        print(f"   Доступно інструментів: {len(mcp_tools)}")

        if mcp_tools:
            print(f"   📋 Інструменти:")
            for tool in mcp_tools:
                print(f"      • {tool.name}")
        print()

        agent = Agent(
            role='Knowledge Manager',
            goal='Зберігати та керувати знаннями через knowledge graph',
            backstory='Ти експерт з організації знань та створення зв\'язків між концепціями.',
            tools=mcp_tools,
            verbose=True
        )

        task = Task(
            description='''
            Використай memory інструменти для:
            1. Створення сутностей: "CrewAI", "MCP", "Python"
            2. Створення зв'язків між ними (наприклад: CrewAI uses MCP)
            3. Збереження факту про проект
            4. Отримання інформації з knowledge graph
            ''',
            agent=agent,
            expected_output='Звіт про створений knowledge graph'
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        print(f"\n✅ Результат: {result}")


# ============================================================================
# MAIN - Запуск прикладів
# ============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     MCP SERVERS EXAMPLES з CrewAI                          ║
║                                                                            ║
║  Демонстрація інтеграції різних офіційних MCP серверів                    ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    examples = {
        "1": ("Sequential Thinking", example_sequential_thinking),
        "2": ("Filesystem", example_filesystem),
        "3": ("Fetch", example_fetch),
        "4": ("Git", example_git),
        "5": ("Memory", example_memory),
    }

    print("Доступні приклади:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Запустити всі приклади")
    print()

    choice = input("Оберіть приклад (0-5) або Enter для виходу: ").strip()

    if choice == "0":
        for name, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Помилка в {name}: {e}\n")
    elif choice in examples:
        name, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
            import traceback
            traceback.print_exc()
    elif choice:
        print("❌ Невірний вибір!")

    print("\n✅ Завершено!")
