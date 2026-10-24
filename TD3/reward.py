import numpy as np

def method1(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2):

    # 基础回报
    r = utility / cost if cost != 0 else 0

    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)

    # 鼓励发数据包，发一个包给一点固定奖励
    r += float(len_send_pks)

    # 判断是否为有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)

    # 施加惩罚
    if no_send_with_resources:
        r = r - lambda_2 * ( (e_t - alpha) / alpha + len_backlog/1000 + len_visible)

    #最后截断一下
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method2(utility, cost, violate, lambda_1, alpha, energy_after,len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u, eta, C_base):

    # 基础回报
    r = utility / (cost + C_base )

    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)
    #最后截断一下
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method3(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2):

    # 基础回报
    r = utility / cost if cost != 0 else 0

    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)

    # 判断是否为有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)

    # 施加惩罚
    if no_send_with_resources:
        r = r - lambda_2 * ( (e_t - alpha) / alpha + len_backlog/1000 + len_visible)

    #最后截断一下
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method4(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2, total_c, total_u):
    # 基础回报
    r = total_u / total_c if total_c != 0 else 0
    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)
    # 最后截断一下
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method5(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2, total_c, total_u):
    # 基础回报
    r = total_u / total_c if total_c != 0 else 0
    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)
    # 判断是否为有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)
    # 施加惩罚
    if no_send_with_resources:
        r = r - lambda_2 * ( abs(e_t - alpha) / alpha + len_backlog/1000 + len_visible)
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method6(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2, total_c, total_u):
    # 基础回报
    r = total_u / total_c if total_c != 0 else 0
    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)
    # 判断是否为有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)
    # 施加惩罚
    r+=len_send_pks
    if no_send_with_resources:
        r = r - lambda_2 * ( abs(e_t - alpha) / alpha + len_backlog/1000 + len_visible)
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method7(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2, total_c, total_u):

    r = total_u / total_c if total_c != 0 else 0

    # 能量中性约束
    r -= float(violate) * lambda_1 * (abs(alpha - energy_after) / alpha)

    # 惩罚有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)
    if no_send_with_resources:
        r = r - lambda_2

    r = float(np.clip(r, -100.0, 100.0))

    return round(r, 3)

def method8(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2, total_c, total_u):

    # 基础回报
    r = total_u / total_c if total_c != 0 else 0

    # 能量中性约束
    r -= lambda_1

    # 判断是否为有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)
    if no_send_with_resources:
        r = r - 2 * lambda_2

    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method9(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2):

    # 基础回报
    r = utility / cost if cost != 0 else 0

    # 能量中性约束
    r -= float(violate) * lambda_1

    # 惩罚有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)
    if no_send_with_resources:
        r = r - lambda_2

    #最后截断一下
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def method10(utility,cost,violate,lambda_1,alpha,energy_after,len_visible,len_backlog,len_send_pks,e_t,e_init,lambda_2, eta):

    # 基础回报
    r = utility - eta * cost

    # 能量中性约束
    r -= float(violate) * lambda_1

    # 惩罚有资源却不发送
    no_send_with_resources = (len_send_pks == 0) and (e_t > alpha) and (len_backlog > 0) and (len_visible > 0)
    if no_send_with_resources:
        r = r - 2 * lambda_2

    #最后截断一下
    r = float(np.clip(r, -100.0, 100.0))
    return round(r, 3)

def _reward(
    method: str,
    utility, cost, violate, lambda_1, alpha, energy_after,
    len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u, eta, C_base
):
    if method == "u/c-lambda1-lambda2+n":
        return method1(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2
        )
    elif method == "u/c-lambda1":
        return method2(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init, lambda_2, total_c, total_u, eta, C_base
        )
    elif method == "u/c-lambda1-lambda2":
        return method3(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2
        )

    elif method == "tu/tc-lambda1":
        return method4(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u
        )
    elif method == "tu/tc-lambda1-lambda2":
        return method5(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u
        )
    elif method == "tu/tc-lambda1-lambda2+n":
        return method6(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u
        )
    elif method == "tu/tc-lambda1-lambda2 * 1":
        return method7(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u
        )
    elif method == "tu/tc-lambda1*1-lambda2*1":
        return method8(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,total_c, total_u
        )
    elif method == "u/c-lambda1*1-lambda2*2":
        return method9(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2
        )
    elif method == "u-nc-lambda1*1-lambda2*2":
        return method10(
            utility, cost, violate, lambda_1, alpha, energy_after,
            len_visible, len_backlog, len_send_pks, e_t, e_init,lambda_2,eta
        )
    raise ValueError(f"Unknown reward method: {method}")
