/** @file
 * @brief 用速度、重力、有限寿命和对象池实现粒子喷泉。
 *
 * 模拟更新与绘制分离：step 改状态，draw 只读取状态。
 * 每步顺序为：运动 → 寿命递减 → 越界回收 → 新粒子发射。
 */
#include "particles.h"
#include <math.h>

/**
 * @brief 推进独立的线性同余随机流；仅用于视觉教学，不用于安全用途。
 * @param system 有效的可读写系统，rng 会更新。
 * @return [0,1) 内的伪随机数；无全局 rand 状态。
 */
static double uniform(ParticleSystem *system)
{
    /* 线性同余递推：新状态 = a * 旧状态 + c，写回 uint32_t 后按 2^32 取模。
     * UINT32_C 指定适合 uint32_t 运算的整数常量；这里不使用全局 rand()。
     */
    system->rng = system->rng * UINT32_C(1664525) + UINT32_C(1013904223);
    /* 丢弃低 8 位，保留高 24 位。其最大值为 2^24 - 1，除以 2^24
     * （16777216）得到 [0,1)，不会生成恰好为 1 的值。
     */
    return (double)(system->rng >> 8) / 16777216.0;
}

/** @copydoc particles_reset */
void particles_reset(ParticleSystem *system, uint32_t seed)
{
    /* 复合字面量创建一个临时结构体值，再整体赋给调用者的对象。
     * 指定初始化只写 rng；其余成员按 C 的聚合初始化规则全部置零，
     * 因而所有 remaining 都为零，无需逐个释放或清理粒子。
     */
    *system = (ParticleSystem){.rng = seed};
}

/** @copydoc particles_step */
void particles_step(ParticleSystem *system)
{
    /* 用 1.0 触发浮点除法；若写成 1 / PARTICLE_HZ，整数结果会是零。
     * dt 是模拟秒数，不随这一帧显示用了多久而变化。
     */
    const double dt = 1.0 / PARTICLE_HZ;
    /* 第一轮只更新已有粒子，使新生粒子不会在出生当步扣除寿命。 */
    for (size_t i = 0; i < PARTICLE_CAPACITY; ++i) {
        /* 取数组元素地址后通过 -> 原地修改，不复制整个粒子结构体。 */
        Particle *particle = &system->pool[i];
        if (particle->remaining == 0) {
            /* 空槽不运动，也不做无符号减法，避免 0 - 1 回绕成大整数。 */
            continue;
        }
        /* 半隐式欧拉：先更新速度，再用新速度更新位置。
         * 重力是 +90 像素/秒²，因为画布 y 轴向下；每步 vy 增加 1.5。
         * 初始 vy 为负，随后趋于零再变正，粒子就会先上升再下落。
         * 水平方向没有加速度，vx 保持不变。这是离散近似，不是解析轨迹。
         */
        particle->vy += 90 * dt;
        particle->x += particle->vx * dt;
        particle->y += particle->vy * dt;
        --particle->remaining;
        /* 合法范围为 [0,宽) × [0,高)，等于宽或高也已越界。
         * 回收只改标记，不必擦除旧字段；第二轮发射会覆盖整个槽位。
         * 寿命刚减到零的槽位也会被第二轮直接视为空闲。
         */
        if (particle->x < 0 || particle->x >= GFX_WIDTH || particle->y < 0 ||
            particle->y >= GFX_HEIGHT) {
            particle->remaining = 0;
        }
    }
    unsigned int emitted = 0;
    /* 第二轮从低下标开始寻找空槽；本步最多发射两个，即每模拟秒 120 个。
     * 若池满，循环不会覆盖任何存活粒子，也不会消耗随机数。
     */
    for (size_t i = 0; i < PARTICLE_CAPACITY && emitted < 2; ++i) {
        if (system->pool[i].remaining != 0) {
            continue;
        }
        /* 将 [0,1) 线性映射到速度区间，随机横向速度形成喷泉的展开角度。 */
        double vx = -35 + 70 * uniform(system);
        double vy = -110 + 40 * uniform(system);
        /* 非负浮点数转为整数时截去小数部分：61*u 得到 0..60，
         * 所以寿命是 90..150 步，而不是 90..149；同理 green 是 180..230。
         */
        unsigned int lifetime = 90 + (unsigned int)(61 * uniform(system));
        uint8_t green = (uint8_t)(180 + (unsigned int)(51 * uniform(system)));
        /* 发射点在画布底部附近；整体赋值替换上次使用该槽位留下的字段。
         * 保存初始 lifetime，后续才能计算剩余寿命占比。
         */
        system->pool[i] = (Particle){.x = 160,
                                     .y = 210,
                                     .vx = vx,
                                     .vy = vy,
                                     .remaining = lifetime,
                                     .lifetime = lifetime,
                                     .green = green};
        ++emitted;
    }
}

/** @copydoc particles_count */
size_t particles_count(const ParticleSystem *system)
{
    size_t count = 0;
    for (size_t i = 0; i < PARTICLE_CAPACITY; ++i) {
        /* C 的比较表达式产生 0 或 1，可直接累加为存活数。
         * 这里扫描统计，不维护额外计数器，避免回收/重置时忘记同步。
         */
        count += system->pool[i].remaining != 0;
    }
    return count;
}

/** @copydoc particles_draw */
void particles_draw(const ParticleSystem *system, GfxCanvas *canvas)
{
    for (size_t i = 0; i < PARTICLE_CAPACITY; ++i) {
        const Particle *particle = &system->pool[i]; /* const 防止绘制意外修改模拟状态。 */
        if (particle->remaining == 0) {
            continue;
        }
        /* 必须先转 double 才做除法，否则整数除法会让绝大多数比例变成零。
         * 存活粒子的 lifetime > 0，所以不会除零；比例在 (0,1] 之间。
         * 例：剩余一半寿命时，(255,200,60) 变为 (127,100,30)。
         */
        double brightness = (double)particle->remaining / particle->lifetime;
        uint32_t color =
            gfx_rgb((uint8_t)(255 * brightness), (uint8_t)(particle->green * brightness),
                    (uint8_t)(60 * brightness));
        /* 模拟保留亚像素精度，只有提交绘图时才取整。
         * true 表示填充；逻辑 2×2 像素在两倍窗口中显示为 4×4 色块。
         * gfx_rect 会裁切跨越右/下边界的部分。未做 alpha 混合，后画覆盖先画。
         */
        gfx_rect(canvas, (int)floor(particle->x), (int)floor(particle->y), 2, 2, true, color);
    }
}
