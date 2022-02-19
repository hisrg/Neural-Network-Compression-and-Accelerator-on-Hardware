# A Comprehensive Survey on Hardware-Aware  Neural Architecture Search
# 硬件感知神经结构搜索的综述
## IV.TAXONOMY OF HW-NAS(HW-NAS的分类)
Unlike conventional NAS, where the goal is to find the best architecture that maximizes model accuracy, hardware-aware NAS (HW-NAS) has multiple goals and multiple views of the problem. We can classify these goals into three categories (See figure 9 from left to right)：

传统的NAS的目标是找到最大化模型准确性的最佳架构，与之不同的是，硬件感知的NAS (HW-NAS)具有多个目标和多个问题视图。我们可以将这些目标分为三类(见图9，从左到右)：
* Single Target, Fixed Configuration : Most of existing HW-NAS fall under this category. The goal is to find the best architecture in terms of accuracy and hardware efficiency for one single target hardware. Consequently, if a new hardware platform has to be used for the NAS, we need to rerun the whole process and feed it the right values to calculate the new hardware s cost. These methods generally define the problem as a constrained or multi-objective optimization problem [53], [47], [23]. Within this category, two approaches are adopted:

* 单目标，固定配置：大多数现有的HW-NAS都属于这一类别。我们的目标是为单个目标硬件找到精度和硬件效率方面的最佳架构。因此，如果必须为NAS使用一个新的硬件平台，我们需要重新运行整个流程，并为它提供正确的值，以计算新硬件的成本。这些方法通常将问题定义为有约束或多目标优化问题[53]，[47]，[23]。在这一类别中，采用了两种方法:

-- Hardware-aware search strategy where the search is defined as a multi-objective optimization problem.While searching for the best architecture, the search
algorithm calls the traditional evaluator component to get the accuracy of the generated architecture but also a special evaluator that measures the hardware
cost metric (e.g., latency, memory usage, energy consumption). Both model accuracy and hardware cost guide the search and enable the NAS to find the most efficient architecture.\
--硬件感知搜索策略，其中搜索被定义为一个多目标优化问题。在寻找最佳架构的过程中，搜索
算法调用传统的评估器组件来获得生成的体系结构的准确性，但也调用一个特殊的评估器来测量硬件
成本度量(例如，延迟，内存使用，能源消耗)。模型准确性和硬件成本都会指导搜索，并使NAS能够找到最有效的体系结构。\
--On the other hand, the Hardware-aware Search Space approach uses a restricted pool of architectures. Before the search, we either measure the operators’ performance on the target platform or we define a set of rules that will refine the search space;
eliminate all the architectures’ operators that do not perform well on the target hardware. For example, HURRICANE [72] uses different operator choices
for three types of mobile processors: Hexagon DSP, ARM CPU and Myriad Vision Processing Unit (VPU). Accumulated domain knowledge from prior experimentation on a given hardware platform help narrow down the search space. For instance, they do not to use depthwise convolutions for CPU, squeeze and excitation mechanisms for VPU and they do
not lower the kernel sizes for a DSP. Such gathered empirical information helps to define three different search spaces according to the targeted hardware
platform. Note that after defining the search space with these constraints, the search strategy is similar to the one used by conventional NAS, which means
that the search is solely based on the accuracy of the architecture and no other hardware metric is incorporated.\

--另一方面，硬件感知的搜索空间方法使用受限的架构池。在搜索之前，我们要么测量操作符在目标平台上的性能，要么定义一组规则来细化搜索空间;消除所有在目标硬件上执行不好的架构操作符。例如，HURRICANE[72]为三种类型的移动处理器使用了不同的运营商选择:Hexagon DSP、ARM CPU和Myriad Vision Processing Unit (VPU)。在给定的硬件平台上积累的领域知识有助于缩小搜索空间。例如，他们没有为CPU使用深度卷积，为VPU使用挤压和激励机制，他们没有降低DSP的内核大小。这些收集到的经验信息有助于根据目标硬件平台定义三个不同的搜索空间。请注意，在用这些约束定义搜索空间之后，搜索策略与传统NAS使用的策略类似，这意味着搜索只基于体系结构的准确性，不包含其他硬件指标。\
* Single Target, Multiple Configurations: the goal of this category is not only to get the most optimal architecture that gets the best accuracy but also to get an
optimal architecture with latency guranteed to meet the target hardware specification. For example, the authors of FNAS [116] define a new hardware search space
containing the different FPGA specifications (e.g., tiling configurations). They also use a performance abstraction model to measure the latency of the searched neural
architectures without doing any training. This allows them to quickly prune architectures that do not meet the target hardware specifications. In [129], the authors
use the same approach for ASICs and define a hardware search space that contains various ASIC templates.\
* 单目标，多配置:这一类别的目标不仅是获得最优的架构，获得最佳的准确性，而且还获得
优化的架构，保证满足目标硬件规格的延迟。例如，FNAS[116]的作者定义了一个新的硬件搜索空间
包含不同的FPGA规格(例如，平铺配置)。他们还使用性能抽象模型来测量搜索神经的延迟
没有经过任何培训的架构。这允许他们快速删除不符合目标硬件规格的架构。在[129]中，作者
对ASIC使用相同的方法，并定义一个包含各种ASIC模板的硬件搜索空间。\

* Multiple Targets: In this third category, the goal is to find the best architecture when given a set of hardware platforms to optimize for. In other words, we try to
find a single model that performs relatively well across different hardware platforms. This approach is the most favourable choice, especially in mobile development as it
provides more portability. This problem was tackled by [127], [128] by defining a multi-hardware search space. The search space contains the intersection of all the
architectures that can be deployed in the different targets. Note that, targeting multiple hardware specifications at once is harder as the best model for a GPU, can be
very different to the best model for a CPU (i.e., for GPUs wider models are more appropriate while for CPUs deeper models are).
* 多目标:在这第三类中，目标是在给定一组硬件时，找到最佳的架构可优化的平台。换句话说，我们试图
找到一个在整体上表现相对良好的单一模型不同的硬件平台。这种方法是最有效的，这也是一个不错的选择，尤其是在移动开发领域提供了更多的可移植性。这个问题被解决了[127]，[128]通过定义一个多硬件搜索空间。搜索空间包含所有的交集可以部署在不同目标中的架构。注意，针对多个硬件规格在对于GPU来说，一次是最好的模型非常不同于CPU的最佳模型(例如，为了更大型号的gpu更适合cpu 更深层次的模型)。\

## V.SEARCH SPACES(搜索空间)
Two different search spaces have been adopted in the literature to define the search strategies used in HW-NAS: the Architecture Search Space and the Hardware Search Space.
文献中采用了两种不同的搜索空间来定义HW-NAS中使用的搜索策略:架构搜索空间和硬件搜索空间。\
A.Architecture Search Space(架构搜索空间)\
The Architecture Search Space is a set of feasible architectures from which we want to find an architecture with  high performance. Generally, it defines a set of basic network operators and how these operators can be connected to construct the computation graph of the model. We distinguish two approaches to design an architecture search space: 
架构搜索空间是一组可行的架构，我们想从中找到一个高性能的架构。它一般定义一组基本的网络算子，以及这些算子之间的连接方式来构造模型的计算图。我们区分了两种设计建筑搜索空间的方法:\
1).Hyperparameter Optimization for a fixed architecture:In this approach a neural architecture is given includingits operator choices. The objective is limited to optimizing the architecture hyperparameters (e.g., number of channels, stride, kernel size).\
--1).固定体系结构的超参数优化:在这种方法中，给出了包括算子选择在内的神经体系结构。目标局限于优化架构超参数(例如，通道数量、步幅、内核大小)。\
2).True Architecture Search Space: The search space allows the optimizer to choose connections between operations and to change the type of operation.
--2)真正的架构搜索空间:搜索空间允许优化器选择操作之间的连接，并更改操作类型。\
Both approaches have their advantages and disadvantages but it is worth mentioning that although former approach reduces the search space size, it requires considerable human
expertise to design the search space and introduces a strong bias. Whereas the latter approach decreases the human bias but considerably increases the search space size and hence the search time.\
这两种方法各有利弊，但值得一提的是，虽然前者减少了搜索空间的大小，但它需要相当多的人类专业知识来设计搜索空间，并引入了强烈的偏见。而后一种方法减少了人为偏差，但大大增加了搜索空间大小，从而增加了搜索时间。\

Generally, in the later approach, we distinguish three types
(See figure 10):
一般来说，在后一种方法中，我们区分三种类型
(见图10):
* Layer-wise Seach Space, where the whole model is generated from a pool of operators. FBNet Search Space[23], for example, consists of a layer-wise search space with a fixed macro architecture which determines the number of layers and dimensions of each layer where the first and last three layers have fixed operators. The remaining layers need to be optimized. \
* 分层搜索空间，其中整个模型是由操作符池生成的。FBNet搜索空间 例如，[23]由一个具有固定宏架构的分层搜索空间组成层数和每一层的尺寸，其中第一层和最后三层有固定的操作符。其余的层需要优化。
* Cell-based Search Space, where the model is constructed from repeating fixed architecture patterns calledblocks or cells. A cell is often a small acyclic graph that
represents some feature transformation. The cell-based approach relies on the observation that many effective handcrafted architectures are designed by repeating a
set of cells. These structures are typically stacked and repeated a number of time to form larger and deeper architectures. This search space focuses on discovering
the architecture of specific cells that can be combined to assemble the entire neural network. Although cell-based search spaces are intuitively efficient to look for the best model in terms of accuracy, they lack flexibility when it comes to hardware specialization [47], [23].
* 基于单元的搜索空间，其中的模型是由称为块或单元的重复固定架构模式构建的。单元格通常是表示某些特征转换的小非循环图。基于单元的方法依赖于这样的观察:许多有效的手工构建的体系结构都是通过重复一组单元来设计的。这些结构通常是堆叠和重复的时间，形成更大和更深的结构。这个搜索空间的重点是发现特定细胞的结构，这些细胞可以组合成整个神经网络。尽管基于单元格的搜索空间可以直观地高效地寻找最佳模型，但在硬件专门化[47]，[23]时，它们缺乏灵活性。
* Hierarchical Search Space, works in 3 steps: First the cells are defined and then bigger blocks containing a defined number of cells are constructed. Finally the whole
model is designed using the generated cells. MNASNet [47] is a good example of this category of search spaces. The authors define a factorized hierarchical search space
that allows more flexibility compared to a cell-based search space. This allows them to reduce the size of the total search space compared to the global search space.
* 层次搜索空间，工作在3个步骤:首先定义单元格，然后构建包含定义数量单元格的更大块。最后，利用生成的单元对整个模型进行设计。MNASNet[47]是这类搜索空间的一个很好的例子。作者定义了一种分解的分层搜索空间，与基于单元格的搜索空间相比，它具有更大的灵活性。与全局搜索空间相比，这允许它们减少总搜索空间的大小。\
* In existing NAS research works, the authors define a macroarchitecture that generally determines the type of networks considered in the search space. When considering CNNs,
the macro architecture is usually identical to the one shown in figure 1. Therefore, many works [53], [23], [47], [127],[130] differ in the number of layers, the set of operations and the possible hyperparameters values. Recently, the scope of network type is changing. For instance, NASCaps [100] changes their macro-architecture to allow the definition of capsules. Capsules network [131] are basically cell-based CNNs where each cell (or capsule) can contain a different CNN architecture.\
在现有的NAS研究工作中，作者定义了一个宏架构，该架构一般决定了在搜索空间中考虑的网络类型。当考虑到CNN,宏体系结构通常与图1所示的体系结构相同。因此，很多作品[53]、[23]、[47]、[127]、[130]在层数、操作集和可能的超参数值上存在差异。最近，网络类型的范围正在发生变化。例如，NASCaps[100]改变了他们的宏观架构，允许定义胶囊。胶囊网络[131]基本上是基于细胞的CNN，其中每个细胞(或胶囊)可以包含不同的CNN结构。\
Other works like [66], [99] focus on transformers and define their macro-architecture as a transformer model. The search consists of finding the number of attention heads and their internal operations. When dealing with the hyperparameters only, the macro architecture can define a variety of network
types. Authors in [75], [105] mix different definitions, transformers + CNN and transformers + RNN respectively. They define a set of hyperparameters that encompasses the predefined parameters for different network types at the same time.\
其他作品如[66]、[99]关注变压器，并将其宏观架构定义为变压器模型。搜索的内容包括找到关注头的数量及其内部运作。当只处理超参数时，宏体系结构可以定义各种网络
类型。[75]、[105]的作者混合了不同的定义，分别是变压器+ CNN和变压器+ RNN。它们定义了一组超参数，同时包含了针对不同网络类型的预定义参数。
Lately, more work [53], [64] have been considering the use of over-parameterized networks (i.e. supernetworks) to speedup the NAS algorithms. These networks consist of adding architectural learnable weights that select the appropriate operator at the right place. Note that these techniques have been applied to transformers as well [132].
最近，更多的工作[53]，[64]已经考虑使用过度参数化网络(即超级网络)来加速NAS算法。这些网络包括添加体系结构的可学习权值，以在正确的位置选择适当的操作符。请注意，这些技术也应用于变压器[132]。

Finally, in some research efforts, the pool of operators/architectures is refined with only the models that are efficient in the targeted hardware [128], [127]. The search space size is considerably reduced by omitting all the architectures that cannot be deployed.
最后，在一些研究中，运营商/架构池仅通过目标硬件中高效的模型进行了细化[128]，[127]。通过省略所有不能部署的架构，搜索空间的大小大大减少。
## Hardware Search Space (HSS)
Some HW-NAS methods include a HSS component which generates different hardware specifications and optimizations by applying different algorithmic transformations to fit the hardware design. This operation is done before evaluating the model. Although the co-exploration is effective, it increases the
search space time complexity significantly. If we take FPGAs as an example, their design space may include: IP instance categories, IP reuse strategies, quantization schemes, parallel factors, data transfer behaviours, tiling parameters, and buffer sizes. It is arguably impossible to consider all these options as part of the search space due to the added search computation
cost. Therefore, many existing strategies limit themselves to only few options.
一些HW-NAS方法包括一个HSS组件，该组件通过应用不同的算法转换来生成不同的硬件规格和优化，以适应硬件设计。此操作在评估模型之前完成。虽然联合勘探是有效的，但它增加了搜索的时空复杂性显著。 以fpga为例，其设计空间可能包括:IP实例类别、IP重用策略、量化方案、并行因素、数据传输行为、平铺参数和缓冲区大小。 由于增加了搜索计算，要将所有这些选项都考虑到搜索空间中是不可能的 成本。 因此，许多现有的策略限制自己只有很少的选择。
Hardware Search Space (HSS) can be further categorized
as follows:

* Parameter-based: The search space is formalized by aset of different parameter configurations. Given a specific data set, FNAS [116] finds the best performing model, along with the optimization parameters needed for it to be deployed in a typical FPGA chip for deep learning. Their HSS consists of four tiling parameters for the convolutions. FNASs [117] extends FNAS by adding more optimization parameters such as loop unrolling. The authors in [118], [119] used a multi-FPGA hardware search space. The search consists of dividing the
architecture into pipeline stages that can be assigned to an FPGA according to its memory and DSP slices, in addition to applying an optimizer that adjusts the tiling parameters. Another example is presented in [120], where
the adopted approach takes the global structure of an FPGA and adds all possible parameters to its hardware search space including the input buffer depth, memory interface width, filter size and ratio of the convolution
engine. [121] searches the internal configuration of an FPGA by generating simultaneously the architecture hyperparameters, the number of processing elements, and the size of the buffer. FPGA/DNN [133] proposes two
components: Auto-DNN which performs hardware-aware DNN model search and Auto-HLS which generates a synthesizable C code of the FPGA accelerator for explored DNNs. Additional code optimizations such as buffer reallocation and loop fusion on the resulting C-code are added to automate the hardware selection.\
基于参数的:搜索空间由一组不同的参数配置来形式化。 给定特定的数据集，FNAS[116]会找到性能最佳的模型，以及将其部署到典型的FPGA芯片进行深度学习所需的优化参数。 它们的HSS由四个卷积平铺参数组成。 FNASs[117]通过添加更多的优化参数，如循环展开，对FNAS进行了扩展。 作者在[118]、[119]中使用了多fpga硬件搜索空间。 搜索包括划分 除了应用优化器来调整平铺参数外，还可以根据内存和DSP切片将其分配到FPGA的流水线阶段。 另一个例子在[120]中 该方法采用了FPGA的全局结构，并将所有可能的参数添加到硬件搜索空间中，包括输入缓冲区深度、内存接口宽度、滤波器大小和卷积比  引擎。 [121]通过同时生成体系结构超参数、处理元素的数量和缓冲区的大小来搜索FPGA的内部配置。 FPGA/DNN[133]提出了两种方法 Auto-DNN，实现硬件感知的DNN模型搜索;Auto-HLS，为已探索的DNN生成可合成的FPGA加速器C代码。 额外的代码优化，如缓冲区重新分配和循环融合的结果c代码被添加到自动化硬件选择。
* Template-based: In this scenario, the search space is defined as a set of pre-configured templates. For example, NASAIC [129] integrates NAS with Application-Specific Integrated Circuits (ASIC). Their hardware search space
includes templates of several existing successful designs. The goal is to find the best model with the different possible parallelizations among all templates. In addition to the the tiling parameters and bandwidth allocation, the authors in [122] define a set of FPGA platforms and the search finds a coupling of the architecture and FPGA platform that fits a set of pre-defined constraints (e.g., max latency 5ms).\
基于模板的:在这个场景中，搜索空间被定义为一组预先配置的模板。例如，NASAIC[129]将NAS与专用集成电路(ASIC)集成在一起。他们的硬件搜索空间
包括几个现有成功设计的模板。目标是在所有模板中找到具有不同可能并行化的最佳模型。除了平铺参数和带宽分配，作者在[122]中定义了一组FPGA平台，搜索发现了一个架构和FPGA平台的耦合，它符合一组预定义的约束(例如，最大延迟5ms)。\
In general, we can classify the targeted hardware platforms into 3 classes focusing on their memory and computation capabilities:Server Processors: this type of hardware can be found in cloud data centers, on premise data centers, edge servers, or supercomputers. They provide abundant computational
resources and can vary from CPUs, GPUs, FPGAs and ASICs. When available, machine learning researchers focus on accuracy. Many NAS works consider looking for the best architecture in these devices without considering
the hardware-constraints. Nevertheless, many HW-NAS works target server processors to speed up the training process and decrease the extensive resources needed to train a DL architecture and use it for inference.\
一般来说，我们可以根据内存和计算能力将目标硬件平台分为3类:服务器处理器:这种类型的硬件可以在云数据中心、前提数据中心、边缘服务器或超级计算机中找到。它们提供了丰富的计算能力
资源，可以从cpu, gpu, fpga和asic。如果可行，机器学习研究人员将重点放在准确性上。许多NAS工作人员考虑在这些设备中寻找最佳的架构，而无需考虑
硬件约束。然而，许多HW-NAS工作的目标服务器处理器，以加快训练过程，并减少训练DL体系结构和使用它进行推理所需的大量资源
* Mobile Devices: With the rise of mobile devices, the
focus has shifted to enable fast and efficient deep learning
on smartphones. As these devices are heavily constrained
with respect to their memory and computational capabilities, the objective of ML researchers shift to assessing the trade-off between accuracy and efficiency. Many HW-NAS algorithms target smartphones including FBNet [23]
and ProxylessNAS [53] (refer to table III). Additionally, because smartphones usually contain system on chips with different types of processors, some research efforts[79] have started to explore ways to take advantage of
these heterogeneous systems.\
移动设备:随着移动设备的兴起，人们开始
重点已经转移到实现快速高效的深度学习
在智能手机。由于这些设备受到严重限制
关于他们的记忆和计算能力，ML研究人员的目标转移到评估准确性和效率之间的权衡。许多HW-NAS算法都针对智能手机，包括FBNet [23]
和ProxylessNAS[53](见表三)。此外，由于智能手机通常包含不同类型处理器的系统芯片，一些研究努力[79]开始探索利用的方法
这些异构系统。\
* Tiny Devices: The strong growth in use of microcontrollers and IoT applications gave rise to TinyML [134]. TinyML refers to all machine learning algorithms dedicated to tiny devices, i.e, capable of on-device inferenceat extremely low power. One relevant HW-NAS method that targets tiny devices is MCUNet [123], which includes an efficient neural architecture search called TinyNAS. TinyNAS optimizes the search space and handles a variety of different constraints (e.g., device, latency, energy, memory) under low search costs. Thanks to the efficient search, MCUNet is the first to achieves >70% ImageNet
top-1 accuracy on an off-the-shelf commercial microcontroller.

微型设备:微控制器和物联网应用的强劲增长催生了TinyML[134]。TinyML指的是所有用于微型设备的机器学习算法，即能够在极低功耗的情况下进行设备推理。一种针对微型设备的相关HW-NAS方法是MCUNet[123]，其中包括一种名为TinyNAS的高效神经架构搜索。TinyNAS优化了搜索空间，并在低搜索成本下处理各种不同的约束条件(例如，设备、延迟、能量、内存)。由于高效的搜索，MCUNet是第一个实现>70% ImageNet
在一个现成的商用微控制器上，精度是最高的。
## Current Hardware-NAS Trends
Figure 11 shows the different types of platforms that have been targeted by HW-NAS in the literature. In total, we have studied 126 original hardware-aware NAS papers. By target, we mean the platform that the architecture is optimized
for. Usually the search algorithm is executed in a powerful machine, but that is not the purpose of our study. ”No Specific Target” means that the HW-NAS incorporates hardware agnostic constraints into the objective function such as the number of parameters or the number of FLOPs. In the figure, the
tag ”Multiple” means multiple types of processing elements have been used in the HW platform. Table III gives the list of references per targeted hardware.
\
图11显示了文献中HW-NAS针对的不同类型的平台。我们总共研究了126篇原始的硬件感知NAS论文。所谓目标，我们指的是架构被优化的平台
对。通常搜索算法是在一个强大的机器中执行的，但这不是我们研究的目的。”无特定目标”意味着HW-NAS将硬件不确定的约束合并到目标函数中，例如参数的数量或FLOPs的数量。在图中
标签“多”表示在HW平台中使用了多种处理元素。表III给出了每个目标硬件的参考列表。
In the figure, we note that the number of research papers targeting GPUs and CPUs has more or less remained constant. However, we can clearly see that FPGAs and ASICs are gaining popularity over the last 3 years. This is consistent with
the increasing number of deep learning edge applications. Two recent interesting works are [127], [128] both of which target multiple hardware platforms at once.\
在图中，我们注意到针对gpu和cpu的研究论文数量基本保持不变。然而，我们可以清楚地看到，fpga和asic在过去3年里越来越受欢迎。这与
深度学习的边缘应用越来越多。最近的两个有趣的作品是[127]，[128]，它们都同时针对多个硬件平台。
In figure 12, we illustrates the different DNN operations that compose the architecture search space. First, we divide the CNN into two groups, standard CNN which only utilizes.a standard convolution and extended CNN which involves
special convolution operations such as the depthwise separable convolution or grouped convolutions. NAS has been mostly dominated by convolutional neural networks as shown in the figure. However, recent works have started explore more operators by incorporating capsule networks [100], transformers
[139], and GANs [94].
在图12中，我们演示了构成体系结构搜索空间的不同DNN操作。首先，我们把CNN分成两组，标准的CNN，只有利用。一个标准的卷积和扩展的CNN，其中包括
特殊的卷积运算，如深度可分离卷积或分组卷积。如图所示，NAS一直主要由卷积神经网络主导。然而，最近的工作已经开始探索更多的运营商，通过合并胶囊网络[100]，变压器
[139]和GANs[94]。
## HARDWARE-AWARE NAS PROBLEM FORMULATION
## 硬件感知NAS问题定义(形成)
Neural Architecture Search (NAS) is the task of finding a well-performing architecture for a given dataset. It is cast as an optimization problem over a set of decisions that define different components of deep neural networks (i.e., layers, hyperparameters). This optimization problem can simply be seen as formulated in equation 2.\
神经体系结构搜索(NAS)的任务是为给定的数据集找到一个性能良好的体系结构。它被视为一个优化问题，需要一系列决策来定义深度神经网络的不同组成部分(即层次、超参数)。这个优化问题可以简单地看成如公式2所示。\
We denote the space of all feasible architectures as A (alsocalled search space). The optimization method is looking for the architecture α that maximizes the performance metric denoted by f for a given dataset δ. In this context, f can simply be the accuracy of the model.\
我们将所有可行体系结构的空间表示为A(也称为搜索空间)。 优化方法是寻找使性能指标最大化的体系结构α 对于给定的数据集δ，用f表示。 在这种情况下，f可以仅仅是模型的准确性。\
Although it is important to find networks that provide high accuracy, these NAS algorithms tend to give complex models that cannot be deployed on many hardware devices. To
overcome this problem, practitioners consider other objectives,
such as the number of model parameters, the number of
floating-point operations, and device-specific statistics like
the latency or the energy consumption of the model. Different formulations were used to incorporate the hardwareaware objectives within the optimization problem of neuralarchitecture search. We classify these approaches into two classes, single and multi-objective optimization. The single objective optimization can be further classified as two-stage or constrained optimization. Similarly, the multi-objective optimization approach can be further classified as single or multi-objective optimizations. Please refer to figure 13 for a summary of these approaches. These 2 classes are further detailed with examples from the literature in the following
sections:\
尽管找到提供高精度的网络很重要，但这些NAS算法往往会给出无法在许多硬件设备上部署的复杂模型。来克服这个问题，从业者会考虑其他目标，如型号参数的个数、数量等浮点运算和特定于设备的统计信息，比如模型的延迟或能量消耗。 在神经结构搜索的优化问题中，使用不同的公式来整合硬件目标。 我们将这些方法分为单目标优化和多目标优化两类。单目标优化又可分为两阶段优化和约束优化。类似地，多目标优化方法可以进一步分为单目标或多目标优化。这些方法的摘要请参见图13。 这两个类是进一步详细的例子，从文献在下面部分:
###  Single-Objective Optimization
###  单目标优化
In this class, the search is realized considering only one
objective to maximize, i.e the accuracy. Most of the existing
work in the literature [47], [53], [78], [122], [120], that tackle
the hardware-aware neural architecture search, try to formulate
the multi-objective optimization problem into a single objective to better apply strategies like reinforcement learning or
gradient-based methods. We can divide this class into two
different approaches: Two-stage optimization and constrained
optimization.\
在这一类中，搜索只考虑一个目标最大化，即准确性。现有的大部分文献[47]，[53]，[78]，[122]，[120]，处理硬件感知的神经架构搜索，尝试表述将多目标优化问题转化为一个单一目标，以更好地应用策略，如强化学习或基于梯度的方法。 我们可以把这个班分成两个班不同的方法:两阶段优化和约束优化。\
