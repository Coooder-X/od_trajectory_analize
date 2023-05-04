<template>
  <div class="feature-view">
    <view-header viewId="E" title="轨迹特征视图"></view-header>
    <div class="feature-view-content">
      
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, Ref, ref, watch } from "vue";
import ViewHeader from "../ViewHeader.vue";

export default defineComponent({
  components: {
    ViewHeader,
  },
  name: "FeatureView",
  props: {
  },
  setup(props) {
    
    
    return {
      
    }
  },
});
</script>

<style scoped>
.feature-view {
  position: relative;
  width: 440px;
  height: 380px;
  top: calc(-200px - 500px - 10px - 380px);
  left: 710px;
  background-color: white;
}

.feature-view-content {
  height: 100%;
}
</style>