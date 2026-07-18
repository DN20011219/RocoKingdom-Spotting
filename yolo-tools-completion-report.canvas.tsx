import { Divider, Grid, H1, H2, Stack, Stat, Table, Text, Callout } from 'qoder/canvas';

export default function YoloToolsCompletionReport() {
  return (
    <Stack gap={20}>
      <H1>YOLO 标注工具 + 公用截图模块</H1>
      <Text tone="secondary">项目: RocoKingdom-Spotting | 完成时间: 2026-07-18</Text>

      <Divider />

      <Grid columns={4} gap={16}>
        <Stat value="3" label="新建文件" />
        <Stat value="~492" label="总代码行数" />
        <Stat value="3" label="核心 API" />
        <Stat value="100%" label="Spec 完成度" />
      </Grid>

      <Divider />

      <H2>新建文件</H2>
      <Table
        headers={['文件路径', '行数', '说明']}
        rows={[
          ['tools/capture.py', '120', '公用截图工具 — 封装窗口查找、截图、批量截帧'],
          ['tools/yolo_tools/__init__.py', '2', 'YOLO 工具集包初始化'],
          ['tools/yolo_tools/yolo_labeler.py', '372', 'YOLO 数据集标注工具 — OpenCV GUI'],
        ]}
      />

      <Divider />

      <H2>tools/capture.py — 公用截图工具</H2>
      <Table
        headers={['API', '签名', '说明']}
        rows={[
          ['find_game_window', '(keyword="洛克王国") → int | None', '查找游戏窗口，等待前台'],
          ['capture_frame', '(hwnd=None) → np.ndarray', '截取一帧游戏画面'],
          ['capture_batch', '(count, output_dir, ...) → List[Path]', '批量截帧并保存'],
        ]}
        rowTone={['success', 'success', 'success']}
      />

      <Divider />

      <H2>tools/yolo_tools/yolo_labeler.py — 标注工具</H2>

      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H2>运行模式</H2>
          <Table
            headers={['模式', '命令示例']}
            rows={[
              ['游戏截图', 'python tools/yolo_tools/yolo_labeler.py --capture --classes pet1 pet2'],
              ['本地图片', 'python tools/yolo_tools/yolo_labeler.py --images ./imgs --classes a b'],
            ]}
          />
        </Stack>

        <Stack gap={8}>
          <H2>快捷键</H2>
          <Table
            headers={['按键', '功能']}
            rows={[
              ['0-9', '选择当前类别'],
              ['Space / D', '下一张图片'],
              ['A', '上一张图片'],
              ['S', '保存当前进度'],
              ['Q / Esc', '保存并退出'],
              ['左键拖拽', '画 bounding box'],
              ['右键点击', '删除最近的标注'],
            ]}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>输出格式</H2>
      <Table
        headers={['文件/目录', '说明']}
        rows={[
          ['datasets/my_dataset/images/', '原始图片 (frame_0001.png, ...)'],
          ['datasets/my_dataset/labels/', 'YOLO 标注 (class_id cx cy w h 归一化坐标)'],
          ['datasets/my_dataset/data.yaml', 'YOLO 配置 (nc + names)，可直接用于训练'],
        ]}
        rowTone={['success', 'success', 'success']}
      />

      <Divider />

      <H2>核心特性</H2>
      <Grid columns={3} gap={12}>
        <Callout tone="success">
          <Text weight="bold">断点续标</Text>
          <Text size="small">自动加载已有 labels，支持中途退出后继续</Text>
        </Callout>
        <Callout tone="success">
          <Text weight="bold">公用截图模块</Text>
          <Text size="small">tools/capture.py 供 debug 和 labeler 共用</Text>
        </Callout>
        <Callout tone="success">
          <Text weight="bold">YOLO 格式输出</Text>
          <Text size="small">data.yaml 自动生成，可直接用于 tools/train/yolo.py</Text>
        </Callout>
      </Grid>

      <Divider />

      <Callout tone="info">
        <Text>所有 Spec 要求已完整实现。工具已验证可正常导入和运行。</Text>
      </Callout>
    </Stack>
  );
}
