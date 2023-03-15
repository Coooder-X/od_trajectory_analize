<template>
  <div class="global-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <div class="global-view-content">
      <div class="file-comp">
        <b>数据集:</b>
        <el-select
          v-model="dataset"
          class="file-select"
          placeholder="选择数据集"
          @change="changeDataSet"
        >
          <el-option
            label="2020年5月份杭州轨迹数据集"
            value="2020年5月份杭州轨迹数据集"
          />
        </el-select>
        <el-table
          cell-class-name="table_cell"
          :data="tableData"
          class="data-table"
          align="center"
          show-overflow-tooltip 
          :show-header="false"
          :stripe="true"
          :border="true"
        >
          <el-table-column prop="name" label="FileName" width="250" />
        </el-table>
      </div>
      <el-divider direction="vertical" class="divider" />
      <div class="info-comp"></div>
    </div>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { defineComponent, computed, onMounted } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import ViewHeader from "../ViewHeader.vue";

export default defineComponent({
  components: {
    ViewHeader,
  },
  name: "GlobalView",
  props: {},
  setup() {
    const store = useStore();
    const dataset: Ref<String | null> = ref("");
    let tableData: Ref<{ name: string; }[]> = ref([]);

    const changeDataSet = () => { //  后面加上逻辑：修改数据集后，才显示 gis 轨迹点
      tableData.value = new Array(20).fill(0).map((_, index) => {
        return { name: `2020年5月${index+1}日杭州市出租车GPS轨迹点数据.h5` };
      });
    }

    return {
      dataset,
      tableData,
      changeDataSet
    };
  },
});
</script>

<style scoped>
.global-view {
  --menu-width: 50px;
  height: 200px;
  width: 850px;
  background-color: white;
}

.global-view-content {
  height: calc(100% - var(--header-height));
  display: flex;
  justify-content: center;
}

.file-comp {
  height: 100%;
  width: 400px;
  padding: 10px;
}

.info-comp {
  height: 100%;
  width: 100%;
}

.file-select {
  margin-left: 10px;
  width: 180px;
}

.data-table {
  height: 100px;
  width: 250px;
  margin-top: 10px;
}

:deep(.el-table) {
  border: 0.5px rgb(231, 231, 231) solid;
  border-radius: 5px;
}

:deep(.el-table .el-table__cell) {
  padding: 3px 0;
}

:deep(.table_cell .cell) {
  white-space: nowrap;
}

.divider {
  height: 90%;
  top: 5%;
  border-left: 3px #909399 solid;
}
</style>
