中文|[英文](README.md)
 
本Application支持运行在Atlas 200 DK或者AI加速云服务器上，实现了对faster-rcnn目标检测网络的推理功能。

## 前提条件<a name="zh-cn_topic_0167429321_section137245294533"></a>

部署此Sample前，需要准备好以下环境：

-   已完成MindSpore Studio的安装。
-   已完成Atlas 200 DK开发者板与MindSpore Studio的连接，交叉编译器的安装，SD卡的制作及基本信息的配置等。

## 软件准备<a name="zh-cn_topic_0167429321_section181111827718"></a>

运行此Sample前，需要按照此章节获取源码包，并进行相关的环境配置。

1.  获取源码包。

    将[https://github.com/Ascend/sample-objectdetection](https://github.com/Ascend/sample-objectdetection)仓中的代码以MindSpore Studio安装用户下载至MindSpore Studio所在Ubuntu服务器的任意目录，例如代码存放路径为：_/home/ascend/sample-objectdetection_。

2.  以MindSpore Studio安装用户登录MindSpore Studio所在Ubuntu服务器，并设置环境变量DDK\_HOME。

    **vim \~/.bashrc**

    执行如下命令在最后一行添加DDK\_HOME及LD\_LIBRARY\_PATH的环境变量。

    **export DDK\_HOME=/home/XXX/tools/che/ddk/ddk**

    **export LD\_LIBRARY\_PATH=$DDK\_HOME/uihost/lib**

    >![](doc/source/img/icon-note.gif) **说明：**   
    >-   XXX为MindSpore Studio安装用户，/home/XXX/tools为DDK默认安装路径。  
    >-   如果此环境变量已经添加，则此步骤可跳过。  

    输入:wq!保存退出。

    执行如下命令使环境变量生效。

    **source \~/.bashrc**


## 部署<a name="zh-cn_topic_0167429321_section3723145213347"></a>

1.  以MindSpore Studio安装用户进入目标检测网络应用代码所在根目录，如 _/home/ascend/sample-objectdetection_。
2.  执行部署脚本，进行工程环境准备，包括公共库的编译与部署、应用的编译与部署等操作,其中Presenter Server用于接收Application发送过来的数据并通过浏览器进行结果展示。

    bash deploy.sh  _host\_ip_ _model\_mode_

    -   _host\_ip_：对于Atlas 200 DK开发者板，即为开发者板的IP地址。对于AI加速云服务器，即为Host的IP地址。
    -   local：若MindSpore Studio所在Ubuntu系统未连接网络，请使用local模式，执行此命令前，需要参考[网络模型及公共代码库下载](#zh-cn_topic_0167429321_section92241245122511)将依赖的公共代码库ezdvpp下载到“sample-objectdetection/script“目录下。
    -   internet：若MindSpore Studio所在Ubuntu系统已连接网络，请使用internet模式，在线下载依赖代码库ezdvpp。

    命令示例：

    **bash deploy.sh 192.168.1.2 internet**

3.  参考[网络模型及公共代码库下载](#zh-cn_topic_0167429321_section92241245122511)将需要使用的离线模型文件与需要推理的图片上传至Host侧任一属组为HwHiAiUser用户的目录。

    例如将模型文件**faster\_rcnn.om**上传到Host侧的“/home/HwHiAiUser/models“目录下。
    
    图片要求如下:

    - 格式：jpg、png、bmp。
    - 输入图片宽度：16px~4096px之间的整数。
    - 输入图片高度：16px~4096px之间的整数。



## 运行<a name="zh-cn_topic_0167429321_section87121843104920"></a>

1.  在MindSpore Studio所在Ubuntu服务器中，以HwHiAiUser用户SSH登录到Host侧。

    **ssh HwHiAiUser@**_host\_ip_

    对于Atlas 200 DK，host\_ip默认为192.168.1.2（USB连接）或者192.168.0.2（NIC连接）。

    对于AI加速云服务器，host\_ip即为当前MindSpore Studio所在服务器的IP地址。

2.  进入faster-rcnn检测网络应用的可执行文件所在路径。

    **cd \~/HIAI\_PROJECTS/ascend\_workspace/objectdetection/out**

3.  执行应用程序。

    执行**run\_object\_detection\_faster\_rcnn.py**脚本会将推理生成的图片保存至指定路径。

    命令示例如下所示：

    **python3 run\_object\_detection\_faster\_rcnn.py -m  _\~/models/faster\_rcnn.om_  -w  _800_  -h  _600_  -i**

    **_./example.jpg_  -o  _./out_  -c _21_**

    -   -m/--model\_path：离线模型路径。
    -   -w/model\_width：模型的输入图片宽度，为16\~4096之间的整数。
    -   -h/model\_height：模型的输入图片高度，为16\~4096之间的整数。
    -   -i/input\_path：输入图片的目录/路径，可以有多个输入。
    -   -o/output\_path：输出图片的目录，默认为当前目录。
    -   -c/output\_categories：faster\_rcnn检测的类别数\(包含背景\)，为2\~32之间的整数，默认为值为21。

    其他详细参数请执行**python3 run\_object\_detection\_faster\_rcnn.py --help**命令参见帮助信息。


## 网络模型及公共代码库下载<a name="zh-cn_topic_0167429321_section92241245122511"></a>

-   网络模型下载

    目标检测网络应用中使用的模型是经过转化后的适配Ascend 310芯片的模型，适配Ascend 310的模型的下载及原始网络模型的下载可参考[表1](#zh-cn_topic_0167429321_table2025054712436)。如果您有更好的模型，欢迎上传到[https://github.com/Ascend/models](https://github.com/Ascend/models)进行分享。

    将模型文件（.om文件）上传到Host侧任一属组为HwHiAiUser用户的目录。

    **表 1**  检测网络应用使用模型

    <a name="zh-cn_topic_0167429321_table2025054712436"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0167429321_row14250184710432"><th class="cellrowborder" valign="top" width="19.53%" id="mcps1.2.5.1.1"><p id="zh-cn_topic_0167429321_p6250154710435"><a name="zh-cn_topic_0167429321_p6250154710435"></a><a name="zh-cn_topic_0167429321_p6250154710435"></a>模型名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="29.970000000000002%" id="mcps1.2.5.1.2"><p id="zh-cn_topic_0167429321_p202504470434"><a name="zh-cn_topic_0167429321_p202504470434"></a><a name="zh-cn_topic_0167429321_p202504470434"></a>模型说明</p>
    </th>
    <th class="cellrowborder" valign="top" width="32.01%" id="mcps1.2.5.1.3"><p id="zh-cn_topic_0167429321_p172511475435"><a name="zh-cn_topic_0167429321_p172511475435"></a><a name="zh-cn_topic_0167429321_p172511475435"></a>模型下载路径</p>
    </th>
    <th class="cellrowborder" valign="top" width="18.490000000000002%" id="mcps1.2.5.1.4"><p id="zh-cn_topic_0167429321_p1625116471432"><a name="zh-cn_topic_0167429321_p1625116471432"></a><a name="zh-cn_topic_0167429321_p1625116471432"></a>原始网络下载地址</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0167429321_row1925111472431"><td class="cellrowborder" valign="top" width="19.53%" headers="mcps1.2.5.1.1 "><p id="zh-cn_topic_0167429321_p52511447194311"><a name="zh-cn_topic_0167429321_p52511447194311"></a><a name="zh-cn_topic_0167429321_p52511447194311"></a>目标检测网络模型</p>
    <p id="zh-cn_topic_0167429321_p32528473439"><a name="zh-cn_topic_0167429321_p32528473439"></a><a name="zh-cn_topic_0167429321_p32528473439"></a>(faster_rcnn.om)</p>
    </td>
    <td class="cellrowborder" valign="top" width="29.970000000000002%" headers="mcps1.2.5.1.2 "><p id="zh-cn_topic_0167429321_p15252247154312"><a name="zh-cn_topic_0167429321_p15252247154312"></a><a name="zh-cn_topic_0167429321_p15252247154312"></a>此模型为<strong id="zh-cn_topic_0167429321_b17252134715438"><a name="zh-cn_topic_0167429321_b17252134715438"></a><a name="zh-cn_topic_0167429321_b17252134715438"></a>检测网络</strong>应用中使用的模型。</p>
    <p id="zh-cn_topic_0167429321_p12521447144318"><a name="zh-cn_topic_0167429321_p12521447144318"></a><a name="zh-cn_topic_0167429321_p12521447144318"></a>是基于Caffe的Faster RCNN模型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="32.01%" headers="mcps1.2.5.1.3 "><p id="zh-cn_topic_0167429321_p8252247164310"><a name="zh-cn_topic_0167429321_p8252247164310"></a><a name="zh-cn_topic_0167429321_p8252247164310"></a>请从<a href="https://github.com/Ascend/models/" target="_blank" rel="noopener noreferrer">https://github.com/Ascend/models/</a>仓的computer_vision/<span>object_detect</span><span>/</span><span>faster_rcnn</span>目录中下载。</p>
    <p id="zh-cn_topic_0167429321_p8252184713434"><a name="zh-cn_topic_0167429321_p8252184713434"></a><a name="zh-cn_topic_0167429321_p8252184713434"></a>对应版本说明请参考当前目录下的<span>README.md</span>文件。</p>
    </td>
    <td class="cellrowborder" valign="top" width="18.490000000000002%" headers="mcps1.2.5.1.4 "><p id="zh-cn_topic_0167429321_p1225244714433"><a name="zh-cn_topic_0167429321_p1225244714433"></a><a name="zh-cn_topic_0167429321_p1225244714433"></a>请从<a href="https://github.com/Ascend/models/" target="_blank" rel="noopener noreferrer">https://github.com/Ascend/models/</a>仓的computer_vision/<span>object_detect</span><span>/</span><span>faster_rcnn</span>目录下的<span>README.md</span>文件获取。</p>
    <p id="zh-cn_topic_0167429321_p192524479435"><a name="zh-cn_topic_0167429321_p192524479435"></a><a name="zh-cn_topic_0167429321_p192524479435"></a></p>
    </td>
    </tr>
    </tbody>
    </table>

-   依赖代码库下载

    将依赖的软件库下载到“sample-objectdetection/script“目录下。

    **表 2**  依赖代码库下载

    <a name="zh-cn_topic_0167429321_table6701646132617"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0167429321_row1970164692615"><th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0167429321_p1470646172612"><a name="zh-cn_topic_0167429321_p1470646172612"></a><a name="zh-cn_topic_0167429321_p1470646172612"></a>模块名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0167429321_p187004619261"><a name="zh-cn_topic_0167429321_p187004619261"></a><a name="zh-cn_topic_0167429321_p187004619261"></a>模块描述</p>
    </th>
    <th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0167429321_p170846112618"><a name="zh-cn_topic_0167429321_p170846112618"></a><a name="zh-cn_topic_0167429321_p170846112618"></a>下载地址</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0167429321_row57014462267"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0167429321_p10701746152618"><a name="zh-cn_topic_0167429321_p10701746152618"></a><a name="zh-cn_topic_0167429321_p10701746152618"></a>EZDVPP</p>
    </td>
    <td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0167429321_p20701846182611"><a name="zh-cn_topic_0167429321_p20701846182611"></a><a name="zh-cn_topic_0167429321_p20701846182611"></a>对DVPP接口进行了封装，提供对图片/视频的处理能力。</p>
    </td>
    <td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0167429321_p770124652612"><a name="zh-cn_topic_0167429321_p770124652612"></a><a name="zh-cn_topic_0167429321_p770124652612"></a><a href="https://github.com/Ascend/sdk-ezdvpp" target="_blank" rel="noopener noreferrer">https://github.com/Ascend/sdk-ezdvpp</a></p>
    <p id="zh-cn_topic_0167429321_p870154612614"><a name="zh-cn_topic_0167429321_p870154612614"></a><a name="zh-cn_topic_0167429321_p870154612614"></a>下载后请保持文件夹名称为ezdvpp。</p>
    </td>
    </tr>
    </tbody>
    </table>


