<template>
  <div class="header">
    <div class="circle"> {{ viewId }} </div>
    <span class="title"> {{ title }} </span>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';

export default defineComponent({
  name: 'ViewHeader',
  components: {},
   props: {
    viewId: String,
    title: String,
  },
});
</script>

<style scoped>
.header {
  width: 100%;
  height: var(--header-height);
  background-color: silver;
}

.circle {
  position: absolute;
  margin: 6px 10px 6px 10px;
  width: 23px;
  height: 23px;
  line-height: 23px;
  background-color: #2c3e50;
  border-radius: 50%;
  color: white;
  text-align: center;
  font-size: 18px;
}

.title {
  margin-left: 40px;
  font-weight: bolder;
  font-size: 20px;
  line-height: 35px;
}

</style>
