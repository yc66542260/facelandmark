
//#include <string.h>
//#include <unistd.h>
//#include <stdlib.h>

#include <rtthread.h>
#include <stdio.h>
#include <rt_ai.h>
#include <rt_ai_log.h>
#include "lcd.h"

#include "face_landmark_region_layer.h"
#include "rt_ai_facelandmark_model.h"
#include "prior.h"

// 重命名AI model的参数
#define MY_MODEL_NAME           RT_AI_FACELANDMARK_MODEL_NAME                //AI model 名字，用于查找系统中已注册的AI模型
#define MY_MODEL_IN_1_SIZE      RT_AI_FACELANDMARK_IN_1_SIZE                 //AI model 的 Input 参数
#define MY_MODEL_OUT_1_SIZE     RT_AI_FACELANDMARK_OUT_1_SIZE                //AI model 的 Output参数

volatile rt_ai_uint32_t g_ai_done_flag = 0;  //判断模型推理一次是否完成
static int ai_done(void *ctx);           //推理完成回调函数
static inline void __rgb565_pixel_reversal(uint8_t *pix, int width, int height)
{
//    uint16_t tmp;
//    for (int i = 0; i < (width*height);i += 2)
//    {
//        tmp = pix[i];
//        pix[i] = pix[i + 1];
//        pix[i + 1] = tmp;
//    }
//
    uint8_t tmp, tmp1;
    for( int i = 0; i < height; i++)
    {
        for(int j = 0; j < width*2; j = j + 4)
        {
            tmp = *(pix + i*width*2 + j);
            tmp1 = *(pix + i*width*2 + j + 1);
            *(pix + i*width*2 + j) = *(pix + i*width*2 + j + 2);
            *(pix + i*width*2 + j + 1) = *(pix + i*width*2 + j + 3);
            *(pix + i*width*2 + j + 2) = tmp;
            *(pix + i*width*2 + j + 3) = tmp1;
        }
    }
}

static void __rgb565_image_vertical_flip(uint8_t *pix, int width, int height)
{
    uint8_t tmp;
    for( int i = 0; i < height/2; i++)
    {
        for(int j = 0; j < width*2; j++)
        {
            tmp = *(pix + i*width*2 + j);
            *(pix + i*width*2 + j) = *(pix + (height - i - 2)*width*2 + j);
            *(pix + (height - i - 2)*width*2 + j) = tmp;
        }
    }
}

static void __rgb888_image_vertical_flip(uint8_t *pix, int width, int height)
{
    uint8_t tmp;
    for( int i = 0; i < height/2; i++)
    {
        for(int j = 0; j < width; j++)
        {
            tmp = *(pix + i*width + j);
            *(pix + i*width + j) = *(pix + (height - i)*width + j);
            *(pix + (height - i)*width + j) = tmp;
        }
    }

    for(int i = 0; i < height/2; i++)
    {
        for(int j = 0; j < width; j++)
        {
            tmp = *(pix + height*width  + i*width + j);
            *(pix + width*height + i*width + j) = *(pix + height*width + (height - i)*width + j);
            *(pix + height*width + (height - i)*width + j) = tmp;
        }
    }

    for(int i = 0; i < height/2; i++)
    {
        for(int j = 0; j < width; j++)
        {
            tmp = *(pix + height*width*2  + i*width + j);
            *(pix + width*height*2 + i*width + j) = *(pix + height*width*2 + (height - i)*width + j);
            *(pix + height*width*2 + (height - i)*width + j) = tmp;
        }
    }
}

int facelandmark_thread_entry(void *params)
{
    /* 初始化相机 */
    uint8_t tmp = 0;
    static uint8_t *cam_buffer;
    static uint8_t *kpu_image;
    static uint8_t *image_layer_1;
    static uint8_t *image_layer_2;
    static uint8_t *image_layer_3;
    cam_buffer = rt_malloc((240 * 320 * 2)+(240 * 320 * 3));
    kpu_image = cam_buffer + (240 * 320 * 2);
    image_layer_1 = kpu_image;
    image_layer_2 = kpu_image+240*320;
    image_layer_3 = kpu_image+240*320*2;
    static facelandmark_region_layer_t rl;
    static facelandmark_box_info_t boxes;
    static float variances[2]= {0.1, 0.2};
    static float *pred_box, *pred_landm, *pred_clses;
    static size_t pred_box_size, pred_landm_size, pred_clses_size;
    rt_ubase_t mb_value = 0;
    rt_ai_t face_landmark = NULL;
    rt_device_t dev = 0;
    rt_err_t result = 0;
    rt_size_t size = 0;

    dev = rt_device_find("gc0308");
    if(!dev)
    {
        rt_kprintf("/n find camera device fail /n");
        return -1;
    }
    result = rt_device_init(dev);
    if(result != 0)
    {
        rt_kprintf("/n init camera device fail /n");
        return -1;
    }
    if(rt_device_open(dev, RT_DEVICE_OFLAG_RDWR) != 0)
    {
        dev = 0;
        return -1;
    }

    /*run AI model*/
    //查找AI model，使用rt_ai_find()
    rt_ai_t mymodel = NULL;                                                       //初始化模型信息
    mymodel = rt_ai_find(MY_MODEL_NAME);                                          //查找已注册模型
    if(!mymodel)
    {
        rt_kprintf("\n AI model not find \n");
        return -1;
    }

    //初始化模型，使用rt_ai_init()
    if (rt_ai_init(mymodel, (rt_ai_buffer_t *)kpu_image) != 0)                     //初始化模型，传入输入数据
    {
        rt_kprintf("\n model init fail \n");
        return -1;
    }
    ai_log("rt_ai_init complete..\n");
    facelandmark_region_layer_init(&rl, anchor, 3160, 4, 5, 1, 320, 240, 0.7, 0.4, variances);
    facelandmark_boxes_info_init(&rl, &boxes, 200);
    ai_log("rt_ai_init complete..\n");

    while(1)
    {
        g_ai_done_flag= 0;
        size = rt_device_read(dev, 0, cam_buffer, 0);
        if (size)
        {
            rt_kprintf("/n camera read fail /n");
            return -1;
        }
        ai_log("rt_ai_init complete..\n");

        __rgb888_image_vertical_flip(kpu_image, 320, 240);

        if(rt_ai_run(mymodel, ai_done, &g_ai_done_flag) != 0)
        {
            ai_log("rtak run fail!\n");
            while (1){}  ;
        }
        while(!g_ai_done_flag){};
        ai_log("rt_ai_init complete..\n");

        pred_box = (float*)rt_ai_output(mymodel,  0);
        pred_landm = (float*)rt_ai_output(mymodel, 1);
        pred_clses = (float*)rt_ai_output(mymodel, 2);
        rl.bbox_input= pred_box;
        rl.landm_input= pred_landm;
        rl.clses_input= pred_clses;
        facelandmark_region_layer_run(&rl, &boxes);

        __rgb565_pixel_reversal(cam_buffer, 320, 240);

        __rgb565_image_vertical_flip(cam_buffer, 320 ,240);

        lcd_draw_picture(0, 0, 320, 240, (rt_ai_uint16_t *)cam_buffer);

        /* run key point detect */
        facelandmark_region_layer_draw_boxes(&boxes, facelandmark_drawboxes);
        facelandmark_boxes_info_reset(&boxes);
    }
}

static int ai_done(void *ctx)
{
    *((uint32_t*)ctx)= 1;
    return 0;
}
