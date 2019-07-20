
from PIL import ImageGrab
import cv2
import glob
import recaptcha_ocr
import time
import threading
from functools import partial
import os,shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import csv
from mouse import click, click_random_point_in_rect, move
from recaptcha_ocr import imgToStr

from recaptcha_detector import ObjectDetector
import SysTrayIcon

from run_exe import run
cnt =0
class RecaptchaSolver():
    def extract_num_from_image_name(self,image_name):
        num =int(image_name[len(self.BASE_PATH+'tile_'): -len('.'+self.EXT)])
        return num
    def __init__(self, size = (500,500), ocr_n = 3000, area_n = 500, loop=False, reload_delay=4, btn_delay=1.5,              seperate = False, show=False,ssd=False,  threshold=0.1,ext='png'):
        #reload_delay까지만 쓰고 거의 안씀
        self.SSD = ssd
        if create_detector:
            
            frame = cv2.imread('icon/recaptcha.png')
            cubic = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
            #미리 예열을 해놔야 됨. ㅇㅇ

            self.detector_mask= ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28', threshold,size)
            self.detector_mask.detect_objects(cubic)
            if self.SSD:
                self.detector_ssd = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
                self.detector_ssd.detect_objects(cubic)
        self.remove_all_in_folder('images')
        self.solved_list = []
        self.d1 = 1
        self.d2 = 1
        self.AREA_N = area_n
        self.RELOAD_DELAY = reload_delay
        self.SIZE = size
        self.SHOW = show
        self.SEPERATE = seperate
        self.LOOP = loop
        self.RELOAD_WORDS = ['위의','조건','새 이','확인']
        self.BTN_DELAY = btn_delay
        self.OCR_N = ocr_n
        self.CLASS_DICT = { '주차':'parking meter', '요금' :'parking meter', '정산기':'parking meter','자풍':'car', '자동':'car', '자량' : 'car' , '차랑':'car', '차량': 'car', '오토바이':'motorcycle', '자전거': 'bicycle', '버스': 'bus', '신호': 'traffic light', '신호등':'traffic light', '소화전':'fire hydrant'}
        self.BASE_PATH = 'C:/opcv/recapcha\\'
        self.THRESHOLD = 0.1
        self.EXT = ext
        self.TEXT_PATH = self.BASE_PATH+'text.'+self.EXT
        self.TILE_PATH = self.BASE_PATH + 'tile.'+self.EXT
        self.TILE_SUM_PATH = self.BASE_PATH + 'tile_sum.'+self.EXT
        ext_paths = glob.glob(self.BASE_PATH +'*.'+self.EXT)
        ext_paths.remove(self.TEXT_PATH)
        ext_paths.remove(self.TILE_PATH)
        ext_paths.remove(self.TILE_SUM_PATH)
        ext_paths.sort(key=self.extract_num_from_image_name )
        self.ROBOT_MENU = (("I'm Not a Robot", 'icon/recaptcha.ico', do_nothing),('Console', 'icon/console.ico', toggle_console))
        self.TEMPLATE_MENU =(("Template Matching", 'icon/opcv.ico', do_nothing),('Console', 'icon/console.ico', toggle_console))
        self.DETECT_MENU = (("Detect", 'icon/tf.ico', do_nothing),('Console', 'icon/console.ico', toggle_console))
        self.IMAGE_PATHS = ext_paths
        self.NOTROBOT_TXT = self.BASE_PATH+'notrobot.txt'
        self.BTN_TXT = self.BASE_PATH + 'btn.txt'
        self.TILE_TXT = self.BASE_PATH + 'tile.txt'
        self.SUCCESS_TXT = self.BASE_PATH + 'is_success.txt'
        self.TILE_RECTS_TXT = self.BASE_PATH +'tile_rects.txt'
        self.NOTROBOT_SCAN = self.BASE_PATH + 'exe/Notrobot_scan.exe'
        self.TILE_SCAN = self.BASE_PATH + 'exe/Tile_scan.exe'
        self.SUCCESS_SCAN = self.BASE_PATH + 'exe/is_success.exe'
        self.CONSOLE_ON = self.BASE_PATH + 'exe/consoleOn.exe'
        self.CONSOLE_OFF = self.BASE_PATH + 'exe/consoleOff.exe'
        self.ON = 1
        self.OFF = 0
        self.CONSOLE_STATUS = self.OFF


    
    def hide_console(self, force = False):

        if not force:
            return
        self.CONSOLE_STATUS = self.OFF
        run(self.CONSOLE_OFF)


    def show_console(self, force = False):
        if not force:
            return
        self.CONSOLE_STATUS = self.ON
        run(self.CONSOLE_ON)


    def combine_masks(self,mask_list, mask_size):
        mask = np.zeros(mask_size,dtype='uint8')
        for i in range(0,len(mask_list)):
            area = cv2.resize(mask_list[i] ,dsize=mask_size, interpolation=cv2.INTER_AREA)
            mask = cv2.bitwise_or(mask, area)
        return mask
    

    def check_point_is_in_mask(self,point, mask):
        x,y =point
        if mask[y][x]==1:
            return True
        return False

    def check_range_has_one(self,h_range,w_range,mask):
        l = w_range.start
        r = w_range.stop 
        t = h_range.start 
        b = h_range.stop
        # show_mask('%d %d'%(l,t),mask[t:b,l:r])
        return mask[t:b,l:r].any()

    def show_mask(self,name, mask):
        if self.SHOW:
            #mask 시각화
            ret, thr = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
            # cv2.imshow(name,thr)
        

    def check_rect_has_one(self, rect,mask):
        x,y,w,h = rect
        # show_mask('%d %d'%(x,y),mask[y:y+h,x:x+w])
        return mask[y:y+h,x:x+w].any()
        
    def check_reload_from_ocr_str(self,ocr_str):
        for word in self.RELOAD_WORDS:
            if word in ocr_str:
                return True
        return False

    def get_class_name_from_ocr_str(self,ocr_str):
        for key in self.CLASS_DICT :
            if key in ocr_str:
                return self.CLASS_DICT[key]
    def get_combined_mask(self,recaptcha_class,mask_size):

        # mask만들기
        mask_list = self.get_mask_list_of_class(recaptcha_class)
        mask = self.combine_masks(mask_list,mask_size)

        return mask


    def merge_masks(self, mask_list, rect_list,mask_size):
        merged_mask = np.zeros(mask_size, dtype='uint8')
        for mask, rect in zip(mask_list, rect_list):
            x,y,w,h = rect
            area = cv2.resize(mask,dsize=(w,h),  interpolation=cv2.INTER_AREA)
            merged_mask[y:y+h, x:x+w] = area
        return merged_mask

    def get_merged_mask_from_imgs(self,imgs, rect_list, recaptcha_class,mask_size ):
        mask_list=[]
        for (x,y,w,h),img in zip(rect_list,imgs):
            cubic = cv2.resize(img, dsize=mask_size, interpolation=cv2.INTER_CUBIC)
            dst = self.detector_mask.detect_objects(cubic)    
            mask= self.get_combined_mask(recaptcha_class,mask_size)
            mask_list.append(mask)

        merged_mask = self.merge_masks(mask_list,rect_list,mask_size)
        return merged_mask

    def load_images(self,image_names):
        img_list = []
        for name in image_names:
            img = cv2.imread(name)
            img_list.append(img)
                
        return img_list

    def get_check_list(self,rect_list,mask):
        check_list =[]
        for rect in rect_list:
            bool_val = self.check_rect_has_one(rect,mask)
            check_list.append(bool_val)
        return check_list



    def get_mask_list_of_class(self,class_name):
        mask_list=[]
        for mask, score, _class in zip(self.detector_mask.output_dict['detection_masks'], self.detector_mask.output_dict['detection_scores'],self.detector_mask.output_dict['detection_classes']):
            name =  self.detector_mask.category_index[_class]['name']
            name = 'bus' if name=='truck' else name #truck은 버스로 통합
            if score > self.THRESHOLD and class_name == name :
                mask_list.append(mask)  

        return mask_list

    def read_txt(self,fname, delimiter=','):
        data = []
        with open(fname) as txt:
            txt_reader = csv.reader(txt, delimiter=',')
            for txt in txt_reader:
                data.append(list(map(int, txt)))
        return data    
    def apply_offset_on_rects(self, offset, rect_list):
        offset_applied_rects = []
        for i in range(0,len(rect_list)):
            x= rect_list[i][0] + offset[0]
            y= rect_list[i][1]+ offset[1]
            w =rect_list[i][2]
            h = rect_list[i][3]
            
            offset_applied_rects.append([x,y,w,h])
        return offset_applied_rects 

    def resize_rects(self,rect_list,ssize,dsize):
        h_ratio = dsize[1]/ssize[1]
        w_ratio =dsize[0]/ssize[0]
        resized_rects = []
        for i in range(0,len(rect_list)):
            x= rect_list[i][0] *w_ratio
            y= rect_list[i][1] *h_ratio
            w =rect_list[i][2] *w_ratio
            h = rect_list[i][3] *h_ratio
            
            resized_rects.append(list(map(int,[x,y,w,h])))
        return resized_rects
            
    def read_txt_until_has_data(self, fname, delay):
        while True:
            data= self.read_txt(fname)
            if len(data) !=0:
                return data
            else:
                time.sleep(delay)
    def try_scan_success(self):
        run(self.SUCCESS_SCAN)
        self.show_console(self.CONSOLE_STATUS)
        data = self.read_txt(self.SUCCESS_TXT)
        code = data[0][0]
        return code
    
    def try_scan_tile(self):
        run(self.TILE_SCAN)
        self.show_console(self.CONSOLE_STATUS)
        data = self.read_txt(self.TILE_RECTS_TXT)
        code = data[0][0]
        if code == -1:
            pass
        elif code in [9,16]:
            self.rect_cnt = code
            self.rect_list = data[1:]
            data =self.read_txt(self.BTN_TXT)
            self.next_btn = data[0]
            self.refresh_btn = data[1]
            self.show_console()
        elif code == 0:
            self.rect_cnt = code
            #refresh는 눌러야 되므로 btn txt를 읽음
            data =self.read_txt(self.BTN_TXT)
            self.next_btn = data[0]
            self.refresh_btn = data[1]
        #code가 -1이 아니라면 True를 내보냄 , 즉 detect되면 True를 반환
        return code != -1
    def get_solved_menu(self):
        #deeptuple값이 빈 튜플이면 do_nothing을
        return (('Solved', 'icon/solved.ico')+ ((deeptuple(self.solved_list) or do_nothing),), )

    def get_robot_menu(self):
        return self.ROBOT_MENU +self.get_solved_menu()
        
    def get_template_menu(self):
        return self.TEMPLATE_MENU + self.get_solved_menu()
        
    def get_detect_menu(self):
        return self.DETECT_MENU + self.get_solved_menu()
        
    def scan_tile(self):
        systray.update_menus(self.get_template_menu())
        while True:
            run(self.TILE_SCAN)
            self.show_console(self.CONSOLE_STATUS)
            data = self.read_txt(self.TILE_RECTS_TXT)
            code = data[0][0]
            print('rect_cnt :',code)
            if code == -1:
                continue
            elif code in [9,16]:
                self.rect_cnt = code
                self.rect_list = data[1:]
                data =self.read_txt(self.BTN_TXT)
                self.next_btn = data[0]
                self.refresh_btn = data[1]
                self.show_console()
                break
            elif code == 0:
                data =self.read_txt(self.BTN_TXT)
                self.next_btn = data[0]
                self.refresh_btn = data[1]
                print("wanna refresh in scan_tile")
                self.click_refresh()
                
    def delay_after_click(self):
        time.sleep(self.BTN_DELAY)

    def click_notrobot(self):
        click(self.notrobot_btn)
        self.delay_after_click()

    def click_next(self):
        click(self.next_btn)
        self.delay_after_click()

    def click_refresh(self):
        # global cnt
        # cnt+=1
        # if cnt == 5:
        #     self.show_console()
        #     exit()

        click(self.refresh_btn)
        self.delay_after_click()

    def scan_notrobot(self):
        
        while True:
            run(self.NOTROBOT_SCAN)
            self.show_console(self.CONSOLE_STATUS)
            data = self.read_txt(self.NOTROBOT_TXT)
            code = data[0][0]
            if code == 1:
                self.notrobot_btn = data[1]
                break
                
        self.show_console()
    def extract_class(self):
        #text 이미지 읽기
        text = cv2.imread(self.TEXT_PATH)

        #ocr 하기
        ocr_str = imgToStr(text, self.OCR_N,self.AREA_N)
        ocr_str += '\n'+imgToStr(text, self.OCR_N)
        
        print(ocr_str)

        #ocr된 string에서 class_name 뽑아내기
        self.recaptcha_class = self.get_class_name_from_ocr_str(ocr_str)
        print(self.recaptcha_class)
        self.reload = self.check_reload_from_ocr_str(ocr_str)
        print('reload:',self.reload)
    def remove_all_in_folder(self,folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    def find_notrobot_and_click(self):
        self.scan_notrobot()
        self.click_notrobot()
    # def set_d1(self):
    #     images = glob.glob("images/image_*.jpg")
    #     max_d1 =0 
    #     for image_name in images:
    #         sub_str = image_name[len("images/image_"):]
    #         sliced_idx = sub_str.find("_")
    #         d1 = int(sub_str[:sliced_idx])
    #         max_d1 = max(d1, max_d1)
        
    #     self.d1 = max_d1+1

             


    def divide_rect_into_equal_parts(self,rect, n):
        offset_x, offset_y, w, h =rect
        equal_rects = []
        sqrt_n = n ** (1/2)
        equal_h = int(h/sqrt_n)
        equal_w = int(w/sqrt_n)
        for y in range(0,h, equal_h):
            for x in range(0,w, equal_w):
                equal_rects.append([offset_x+x, offset_y+y,equal_w, equal_h])
                
        return equal_rects

    def create_rects(self):
    

        #tile 읽기

        if self.rect_cnt == 16:
            #16인 경우 tile_sum을 읽어야 함
            self.tile =  cv2.imread(self.TILE_SUM_PATH)
        else:
            self.tile = cv2.imread(self.TILE_PATH)
            data = self.read_txt(self.TILE_TXT)
            tile_tl = data[0]
            offset = [-tile_tl[0], -tile_tl[1]]
            
        #tile size
        h,w,_ = self.tile.shape
        tile_size = (w,h)

        if self.rect_cnt==16:
            ##16인 경우 tile_sum을 읽어온것이므로 
            # converted_rects를 self.rect_list로부터가 아니라 tile_size로 부터 알아내야함

            #rect n등분 하기
            divided_rects = self.divide_rect_into_equal_parts([0,0,w,h], self.rect_cnt)
            # rint(divided_rects)

            #rect 를 resize
            resized_rects = self.resize_rects(divided_rects, tile_size, self.SIZE)
            self.converted_rects = resized_rects
            # rint(resized_rects)

        else:
            ##9인 경우 tile을 읽어온것이므로 
            # converted_rects를 self.rect_list로부터가 알아내야함.
            #             
            #rect에 offset적용
            offset_applied_rects =self.apply_offset_on_rects(offset, self.rect_list)
            # rint(offset_applied_rects)

            #rect 를 resize
            resized_rects = self.resize_rects(offset_applied_rects, tile_size, self.SIZE)
            self.converted_rects = resized_rects
            # rint(resized_rects)

    def create_check_list(self, reloaded=False): 

        #cubic 후 detect_object하기
        cubic = cv2.resize(self.tile, dsize=self.SIZE, interpolation=cv2.INTER_CUBIC)
        object_name = 'images/image_%d_%d_object.jpg' % (self.d1, self.d2)
        image_name = 'images/image_%d_%d_image.jpg'% (self.d1, self.d2)
        mask_name = 'images/image_%d_%d_mask.jpg' % (self.d1, self.d2)
        cv2.imwrite( image_name, cubic)
        self.dst = self.detector_mask.detect_objects(cubic,True)
        cv2.imwrite(object_name, self.dst)
        
        #combined_mask 생성하기
        combined_mask = self.get_combined_mask(self.recaptcha_class, self.SIZE)




        #rect_cnt만큼만 tile_?.png를 불러옴
        tile_list = self.load_images(self.IMAGE_PATHS[:self.rect_cnt])        


        #check list를 만듦
        check_list = self.get_check_list(self.converted_rects, combined_mask)
        print(check_list)
        # mask 시각화
        self.show_mask('combined_mask',combined_mask)
        
        if self.SEPERATE and self.rect_cnt == 9:
            excepted_list=[]
            if self.SSD:
            ## 최적화 가능
                self.check_list = []
                for i in range(0,self.rect_cnt):
                    if check_list[i]:
                        self.check_list.append(True)
                    else:#리로드 되지 않았거나, self.reloaded_indexes에 i가 없으면 체크함
                        if  not reloaded or i in self.reloaded_indexes:
                            print(i+1,end=', ')
                            bool_val = self.check_class_is_in_img(self.recaptcha_class,tile_list[i],self.SIZE,i)
                            self.check_list.append(bool_val)
                        else:
                            excepted_list.append(i+1)
                            pass
                print('만 체크했습니다')
                
            else:
                
                all_zero_rects = []
                all_zero_tiles = []
                for i in range(0,self.rect_cnt):
                    if check_list[i]:
                        pass
                    else:
                        if not reloaded or i in self.reloaded_indexes:
                            print(i+1, end=', ')
                            all_zero_rects.append(self.converted_rects[i])
                            all_zero_tiles.append(tile_list[i])
                        else:
                            excepted_list.append(i+1)
                            pass
                # rint('rects len :', len(self.converted_rects), 'all zero rects len :', len(all_zero_rects))
                # rint('tiles len :', len(tile_list), 'all zero tiles len :', len(all_zero_tiles))
                print('만 체크했습니다')

                merged_mask = self.get_merged_mask_from_imgs(all_zero_tiles, all_zero_rects, self.recaptcha_class, self.SIZE)
                self.show_mask('merged_mask',merged_mask)

                # 최종병합
                final_mask = cv2.bitwise_or(combined_mask,merged_mask)
                self.show_mask('final_mask', final_mask)

                self.check_list = self.get_check_list(self.converted_rects,final_mask)
            if reloaded:
                print(excepted_list,'는 리로드 전에도 인식되지 않았으므로 제외하였습니다.')
            print(self.check_list)
        else:
            final_mask = combined_mask.copy()
            self.check_list = check_list
        
        if(len(self.solved_list) < self.d1):
            self.solved_list.append([ '%d %.1f"' % (self.d1, self.sum_time+ time.time()- self.time), None, [] ])
        else:
            self.solved_list[self.d1-1][0] = '%d %.1f"' % (self.d1, self.sum_time+time.time()- self.time)

        image_menu = ['Image', None, partial(read_and_show, image_name) ]
        object_menu =['Object', None, partial(read_and_show, object_name)  ]
        
        if self.SEPERATE and self.rect_cnt == 9:
            combined_name = 'images/image_%d_%d_combined.jpg' % (self.d1, self.d2)
            ret, thr = cv2.threshold(combined_mask,0,255,cv2.THRESH_BINARY)
            cv2.imwrite(combined_name, thr)
            combined_menu = ['Whole Mask', None, partial(read_and_show, combined_name)  ]

            merged_name = 'images/image_%d_%d_merged.jpg' % (self.d1, self.d2)
            ret, thr = cv2.threshold(merged_mask,0,255,cv2.THRESH_BINARY)
            cv2.imwrite(merged_name, thr)
            merged_menu = ['Merged Mask', None, partial(read_and_show, merged_name)  ]

            ret, thr = cv2.threshold(final_mask,0,255,cv2.THRESH_BINARY)
            cv2.imwrite(mask_name, thr)
            mask_menu =   ['Final Mask', None, partial(read_and_show, mask_name)  ]

            menu_list = [image_menu, object_menu, combined_menu, merged_menu, mask_menu]
        else:
            ret, thr = cv2.threshold(final_mask,0,255,cv2.THRESH_BINARY)
            cv2.imwrite(mask_name, thr)
            mask_menu =   ['Mask', None, partial(read_and_show, mask_name)  ]

            menu_list = [image_menu, object_menu, mask_menu]
        
        solved_sub_menu = ['%d %s' % (self.d2, self.recaptcha_class) , None, menu_list]
        self.solved_list[self.d1-1][2].append(solved_sub_menu)
        

        systray.update_menus(self.get_detect_menu())
        self.d2+=1
    

    def check_class_is_in_img(self, class_name, img, dsize, n=0):
        cubic = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        dst = self.detector_ssd.detect_objects(cubic)
        # cv2.imshow(str(n), dst)
        for _class, score in zip(self.detector_ssd.output_dict['detection_classes'], self.detector_ssd.output_dict['detection_scores']):
            name =  self.detector_ssd.category_index[_class]['name']
            name = 'bus' if name=='truck' else name #truck은 버스로 통합
            if score > self.THRESHOLD and class_name == name:#bus를 찾고싶은경우 truck도 포함함ㄴ
                return True
        return False  
            

    def click_rects(self):
        #converted_rects와 mask에 대한 check_list를 만듦
        self.reloaded_indexes = []
        # rint(self.rect_list)
        #check_list 중 True인 것과 대응되는 rect를 찾아 클릭함
        if not any(self.check_list):
            return
        for i in range(0,self.rect_cnt):
            if self.check_list[i]:
                print(i+1,end=', ')
                self.reloaded_indexes.append(i)
                click_random_point_in_rect(self.rect_list[i])
        self.delay_after_click()
        print('을 클릭했습니다.')
        if self.reload:
            print('delay after click rects :', self.RELOAD_DELAY,'seconds')
            time.sleep(self.RELOAD_DELAY)


            
    
         
    def next_step(self,step):
        # success라면 6의 다음 step인 7을
        # tile이라면 
        print('%d: %f' % (step, time.time()- self.time))
        if step != 1:
            #notrobot 클릭 후 부터 측정함.
            self.sum_time += time.time()- self.time
        if step == 3:
            # ocr에서 class 추출후
            if self.recaptcha_class is None:
                #detect 불가능한 object이거나 ocr이 제대로 이행되지 못한경우 refresh 후 다시 이단계를 재시작합니다.
                print('wanna refresh after step 3')
                self.click_refresh()
                self.solve(2)
            else:
                self.solve(4)
        elif step ==6:
            if self.reload and any(self.check_list):
                #리로드 되는 검사이고 하나라도 클릭했었다면
                #scan_tile후 text, tile box 부터 시작합니다.
                ## 최적화 가능
                self.scan_tile()
                self.create_rects()
                self.solve(5,True)
            else:
                self.solve(7)

        elif step == 7 :
            #확인또는 건너뛰기 버튼 클릭 후 성공 여부를 봐야함.
            
            systray.update_menus(self.get_template_menu())
            while True:
                if self.try_scan_success():
                    following_step = 8
                    break
                elif self.try_scan_tile():
                    following_step = 2
                    break
            if following_step == 8:
                #step이 8이라면 끝이므로 더이상 재귀호출 하지않음
                print("총 걸린시간 : %.1f" % (self.sum_time))
                
                if self.LOOP:
                    self.solved_list[self.d1-1][0] = '%d %.1f"' % (self.d1, self.sum_time)
                    self.d1 +=1
                    self.d2 = 1
                    self.solve()
                else:
                    pass
                
            elif following_step == 2:
                if self.rect_cnt in [9,16]:
                    print('성공은 못했지만 다시 시도해주세요는 안떳음')
                    self.solve(3)
                else:
                    #빨간글씨가 추가된 것 같으므로므로 refresh 후 다시 2단계부터시작
                    print('wanna refresh after step 7')
                    self.click_refresh()
                self.solve(following_step)
        elif step==1:
            
            systray.update_menus(self.get_template_menu())
            while True:
                if self.try_scan_success():
                    following_step = 8
                    break
                elif self.try_scan_tile():
                    #step 7과 달리 rect_cnt가 0이 나와도 계속 돔, 그래서 2를 건너뛰고 3으로 건너뛸 수 있음
                    if self.rect_cnt in [9,16]:
                        following_step = 3
                        break
            if following_step == 8:
                #step이 8이라면 끝이므로 더이상 재귀호출 하지않음
                print("와! 바로성공")
                print("총 걸린시간 : %.1f" % (self.sum_time))
                if self.LOOP:
                    self.solve()
                else:
                    pass
                
            elif following_step == 3:
                
                self.solve(following_step)
        else:
            self.solve(step+1)

    def solve(self,step=1,reloaded=False):
        self.time = time.time()
        
        if step==1:
         ## 1번째 단계 notrobot 박스를 스캔하여 찾아내고 click합니다.
            systray.update_menus(self.get_robot_menu())
            self.sum_time =0
            self.find_notrobot_and_click()
        elif step ==2: 
           ## 2번째단계  -- tile을 스캔합니다.  
          
            self.scan_tile()

        elif step ==3: 
           ## 2번째단계  -- text.png를 읽어와서 recaptcha_class 를 알아냅니다    
            self.extract_class()

        elif step ==4:   
       ## 4번째 단계 tile 및 tile? 이미지들을 읽어서, rect_list를 만듭니다.
            self.create_rects()
        
        elif step ==5:      
            ## 5번째 단계 -- tile.png를 읽고 combined_mask와 그에 따른 check_list를 생성
           # seperate인경우
            # ssd가 아닌경우
            # check값이 True가 아닌 rect와 img들을 all_zero_rects와  all_zero_imgs로 만들고
            # all_zero들로부터 merged_mask를 생성후 combined_mask와 병합하여 final_mask를 만들고 이에 따른 check_list를 생성
            # ssd 인 경우
            # check값이 True가 아닌 img에 대해 check_class_is_in_img를 실행하여 check_list를 갱신함.
            systray.update_menus(self.get_detect_menu())
            self.create_check_list(reloaded)

        elif step ==6:
            ##6번째 단계
            # checklist가 True인 rect 내의 임의의 점을 클릭한다
            self.click_rects()

        elif step == 7:
            #확인 또는 건너뛰기 버튼을 클릭합니다.
            self.click_next()

            
            
        
        
        self.next_step(step)



def deeptuple(l):
    return tuple(map(deeptuple, l)) if isinstance(l, (list, tuple)) else l



def read_and_show( fname,sysTrayIcon):
    cv2.imshow(fname, cv2.imread(fname))
create_detector = True
do_exit=2


if __name__ == '__main__':
    ssd = False #ssd를 사용하는지 여부
    show = False #show_mask을 하는 지 여부
    seperate = True # tile개별 detect하는 지 여부
    loop = True #백그라운드에서 계속해서 돌아가게 할건지
    btn_delay = 1.5
    reload_delay = 2.5
    size = (500,500) # 어떤 size로 고정시켜 detect시킬지 여부 너무작으면 detect가 안되고 너무 크면 오래걸림
    area_n = 500 # imgTostr에서 area_n
    ocr_n = 3000  #3000하면 1초, 10000만 하면 4초
    

    def toggle_console(sysTrayIcon):
        if recaptcha_solver.CONSOLE_STATUS == recaptcha_solver.ON:
            recaptcha_solver.CONSOLE_STATUS = recaptcha_solver.OFF
            recaptcha_solver.hide_console(True)
        else:
            recaptcha_solver.CONSOLE_STATUS = recaptcha_solver.ON
            recaptcha_solver.show_console(True)
    
    def bye(sysTrayIcon): 
        recaptcha_solver.show_console(True)
        cv2.destroyAllWindows()
        print('Bye, then.')

    def do_nothing(sysTrayIcon):
        pass

    hover_text = "Recaptcha Solver"
    

    recaptcha_solver = RecaptchaSolver(size, ocr_n, area_n, loop,reload_delay ,btn_delay, seperate)
    # for i in range(0,9):
    #     frame = cv2.imread('C:/Users/YASUO/Desktop/aaa/tile_{}.png'.format(i))
    #     cubic = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
    #     dst = recaptcha_solver.detector_mask.detect_objects(cubic,True)
    #     cv2.imwrite("C:/Users/YASUO/Desktop/aaa/tile_{}_detect.jpg".format(i), dst)
    menu_options = recaptcha_solver.get_robot_menu()
    systray = SysTrayIcon.SysTrayIcon('icon/solver.ico', hover_text, menu_options, on_quit=bye, default_menu_index=0)
    t = threading.Thread(target=recaptcha_solver.solve)
    t.daemon = True
    t.start()
    systray.start()
    


    # q= False
    # flag = False
    # while True:
    #     key = cv2.waitKey(1) 
    #     if key == ord("q"):
    #         q= True
    #         break
    #     elif key > 0:
    #         break
    # if q:
    #     break
        
