import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from answering_routing import respond

TELEGRAM_BOT_TOKEN = '7738986713:AAGd1KRjqzUkSRBuwW6PL--S89AoVec1baU'

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Hi there! I'm your chatbot. Send me a message and I'll reply based on my training.")

async def handle_message(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    response = respond(user_text)
    await update.message.reply_text(response)

def main() -> None:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Listening for messages...")
    application.run_polling()

if __name__ == '__main__':
    main()
