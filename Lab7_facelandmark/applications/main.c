/*
 * Copyright (c) 2006-2021, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2018/09/30     Bernard      The first version
 * 2022/11/17     YeC          Update the AI Demo with bsp v1.0.10
 */

#include <rtthread.h>            //rtthread OS系统头文件，包含对外API接口
#include <stdio.h>               //C标准库
#include "sysctl.h"              //Kendryte 芯片相关
#include "rt_ai.h"               //rt_ak 平台相关API接口
#include "lcd.h"

/***********************************************
 * for userapps coding, please update this
 ***********************************************/
#include "logo_image.h"
#include "demo_face_landmark.h"

// 屏幕相关参数
#define LCD_WIDTH 320
#define LCD_HEIGHT 240
static struct rt_device_rect_info info;

// 线程相关
#define THREAD_PRIORITY 25
#define THREAD_STACK_SIZE 10192
#define THREAD_TIMESCALE  500

static int facelandmark_thread(void);

/*************************************
 *  main function
 **************************************/
int main(void)
{
    rt_kprintf("Hello RT-Thread!\r\n");                                           //打印
    rt_kprintf("Demo: Mnist\r\n");                                                //打印

    /* Set KPU clock */
    sysctl_clock_enable(SYSCTL_CLOCK_AI);                                         //KPU时钟初始化

    /* init LCD */
    rt_device_t dev = 0;
    rt_err_t result = 0;
    dev = rt_device_find("lcd");
    if(!dev)
    {
        rt_kprintf("/n init lcd device fail /n");
        return -1;
    }

    if(rt_device_open(dev, 0) != 0)
    {
        dev = 0;
        return -1;
    }

    result = rt_device_control(dev, RTGRAPHIC_CTRL_GET_INFO, &info);
    if (result != 0)
   {
       rt_device_close(dev);
       dev = 0;
       return result;
   }
   rt_kprintf(" \n init LCD successfully \n");

   lcd_draw_picture(0, 0, LCD_WIDTH, LCD_HEIGHT, (rt_ai_uint16_t *)LOGO_IMAGE);
   lcd_draw_string(40, 40, "Hello RT-Thread!", RED);
   lcd_draw_string(40, 60, "Demo: Facelankmark", BLUE);

   // 启动facelandmark线程
   facelandmark_thread();

}

static int facelandmark_thread(void)
{
    rt_thread_t result = 0;
    result = rt_thread_create("facelandmark",
            facelandmark_thread_entry,
            RT_NULL,
            THREAD_STACK_SIZE,
            THREAD_PRIORITY,
            THREAD_TIMESCALE);

    /* 如果获得线程控制块，启动这个线程 */
    if (result != RT_NULL)
        rt_thread_startup(result);
}

