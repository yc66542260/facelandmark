from machine import LCD     # �� machine ���� LCD ��
import time
lcd = LCD()                 # ����һ�� lcd ����
lcd.light(False)            # �رձ���
lcd.light(True)             # �򿪱���
lcd.fill(lcd.BLACK)         # ������ LCD ���Ϊ��ɫ
time.sleep(1)
lcd.fill(lcd.RED)           # ������ LCD ���Ϊ��ɫ
time.sleep(1)
lcd.fill(lcd.GREEN)          # ������ LCD ���Ϊ��ɫ
time.sleep(1)
lcd.fill(lcd.BLUE)         # ������ LCD ���Ϊ��ɫ
time.sleep(1)
lcd.fill(lcd.WHITE)         # ������ LCD ���Ϊ��ɫ
lcd.pixel(50, 50, lcd.BLUE) # ����50,50��λ�õ��������Ϊ��ɫ
lcd.text("hello RT-Thread", 0, 0, 16)   # �ڣ�0, 0�� λ���� 16 �ֺŴ�ӡ�ַ���
lcd.text("hello RT-Thread", 0, 16, 24)  # �ڣ�0, 16��λ���� 24 �ֺŴ�ӡ�ַ���
lcd.text("hello RT-Thread", 0, 48, 32)  # �ڣ�0, 48��λ���� 32 �ֺŴ�ӡ�ַ���
lcd.line(0, 50, 239, 50)    # ����㣨0��50�����յ㣨239��50����һ����
lcd.line(0, 50, 239, 50)    # ����㣨0��50�����յ㣨239��50����һ����
lcd.rectangle(100, 100, 200, 200) # �����Ͻ�Ϊ��100,100�������½ǣ�200,200��������
lcd.circle(150, 150, 80)    # ��Բ��λ�ã�150,150�����뾶Ϊ 80 ��Բ
lcd.show_bmp(180, 50, "sun.bmp")  # ��λ�ã�180,50��ΪͼƬ���½�������ʾ�ļ�ϵͳ�е� bmp ͼƬ "sun.bmp"
