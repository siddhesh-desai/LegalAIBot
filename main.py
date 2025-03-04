from LegalAIBot.LegalAIBot import LegalAIBot
from dotenv import load_dotenv

load_dotenv()

bot = LegalAIBot()
print(bot.generate_output("What is NBFC?"))
print("=====================================")
print(bot.generate_output("What is the process of filing a patent?")[0])
