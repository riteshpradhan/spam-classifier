#!/usr/bin/python
import mailbox, os

spam_file = raw_input('Which mailbox file do you want to read from? ')

mbox = mailbox.mbox(spam_file)
message_count = 0

for message in mbox:
    print '==============Subject========\n' + message['subject']
    print '===============Body==========\n'
    if message.is_multipart():
        print ''.join(part.get_payload(decode=True) for part in message.get_payload())
    else:
         print message.get_payload(decode=True)
    message_count += 1
    next_msg = raw_input('Press enter to continue...')
    os.system('clear')

print 'Total messages: ' + str(message_count)

