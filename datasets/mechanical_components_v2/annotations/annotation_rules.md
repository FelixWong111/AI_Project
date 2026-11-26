原始人工JSON为标注权威；原图像素xyxy左上原点，忽略实例不参与计数。ID0为类型或尺寸不符目录。GT由人工对象推导，expected_counts独立固定为各1。D18/F21以JSON为准，整批人工核实见原冻结确认记录。歧义/遮挡无法判定必须REVIEW，不补造框；保留原ignore字段。

坐标以连续像素边界xyxy表示，右/下边界允许等于W/H，面积=(xmax-xmin)*(ymax-ymin)，不加1。YOLO中心/宽高按同一边界转换。GT行annotation_notes保留导出时说明；后续整批人工确认见manifests/human_confirmation.json，不回写P03。
