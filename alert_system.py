import smtplib

def send_alert():

    sender_email = "youremail@gmail.com"
    receiver_email = "youremail@gmail.com"
    password = "your_app_password"

    message = """Subject: SSD Failure Alert

    Warning: SSD Failure Risk Detected!
    Please backup your data immediately.
    """

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    server.login(sender_email, password)

    server.sendmail(sender_email, receiver_email, message)

    server.quit()